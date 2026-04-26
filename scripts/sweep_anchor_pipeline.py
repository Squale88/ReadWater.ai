"""Run the v3 + coord-gen pipeline across every GT cell, score, and render visual overlays.

Reporting is split into two independent reads, per discussion:

  1. IDENTIFICATION COVERAGE (v3 only).  For each GT anchor, find the best
     v3 anchor candidate by compass-token overlap on `position_in_zone` +
     `rationale` vs the GT label + description. No pixel data used. Answers:
     "did v3 see a structure in this region of the cell at all?" Compass
     overlap is the discrimination signal; structure_type compatibility is a
     tiebreaker, not a gate. Labels do not have to match.

  2. PLACEMENT ACCURACY (coord-gen only, given v3 found something).  For
     v3 anchors that mapped to a GT in step 1, compute the pixel and meter
     distance between coord-gen's predicted center and the GT center.
     Answers: "for the structures v3 actually identified, did coord-gen pin
     them to the right pixel?"

Plus a per-cell PNG overlay (data/.../coord_runs/<run>/review/<cell>.png)
showing GT bboxes (red) + GT centers, pipeline bboxes (blue) + pipeline
centers, with thin connector lines from each GT center to its matched
pipeline center. The overlay is the primary review artifact for the human
in the loop.

Per the GT-usage policy in `ground_truth/anchors/Purpose of ground truth.txt`:
T1+T2 are authoritative for coverage; T3 misses are noted, not failures;
pipeline-only anchors (not best-match for any GT) are positive signals, not
false positives.

Usage:
  python scripts/sweep_anchor_pipeline.py                     # run live (grid mode)
  python scripts/sweep_anchor_pipeline.py --grid-mode nogrid  # nogrid v3
  python scripts/sweep_anchor_pipeline.py --replay            # re-score from disk
  python scripts/sweep_anchor_pipeline.py --only root-10-8 root-7-14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Bootstrap .env (so --replay-only invocations work without external setup)
_env = REPO_ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if not os.environ.get(_k):
            os.environ[_k] = _v

import tune_anchor_coords as coord_h  # noqa: E402
import tune_anchor_identification_v3 as v3_h  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep")

# Cells with ground truth on disk. Discovered from gt_anchors.json presence.
def _discover_cells_with_gt() -> list[str]:
    img_dir = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images" / "structures"
    if not img_dir.exists():
        return []
    cells: list[str] = []
    for child in sorted(img_dir.iterdir()):
        if (child / "gt_anchors.json").exists():
            cells.append(child.name)
    return cells


# ------------------------------------------------------------------
# Compass-token matching (identification coverage)
# ------------------------------------------------------------------

# Compass / position vocabulary used by both GT prose and the v3 prompt's
# `position_in_zone` / `rationale`. Cardinal+ordinal words are weighted higher
# because they are more discriminating (every cell has a "center"; only one
# region is genuinely "NW").
_COMPASS_HIGH = {
    "n", "s", "e", "w",
    "ne", "nw", "se", "sw",
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
}
_COMPASS_LOW = {
    "upper", "lower", "top", "bottom",
    "center", "central", "middle", "mid",
    "edge", "corner", "interior",
}
_TOKEN_RE = re.compile(r"[a-z]+")


def _extract_compass(text: str) -> tuple[set[str], set[str]]:
    text = (text or "").lower().replace("-", " ").replace("/", " ")
    words = set(_TOKEN_RE.findall(text))
    return (words & _COMPASS_HIGH), (words & _COMPASS_LOW)


def _compass_score(gt_text: str, v3_text: str) -> tuple[int, set[str]]:
    """Weighted token overlap. Returns (score, shared_tokens)."""
    gt_high, gt_low = _extract_compass(gt_text)
    v3_high, v3_low = _extract_compass(v3_text)
    high_overlap = gt_high & v3_high
    low_overlap = gt_low & v3_low
    score = 2 * len(high_overlap) + len(low_overlap)
    return score, (high_overlap | low_overlap)


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def _haversine_m(lat1, lon1, lat2, lon2):
    import math as _math
    R = 6371000.0
    p1, p2 = _math.radians(lat1), _math.radians(lat2)
    dp = _math.radians(lat2 - lat1)
    dl = _math.radians(lon2 - lon1)
    a = _math.sin(dp / 2) ** 2 + _math.cos(p1) * _math.cos(p2) * _math.sin(dl / 2) ** 2
    return 2 * R * _math.asin(_math.sqrt(a))


def _meters_per_pixel_at(z16_center_lat: float) -> float:
    from readwater.pipeline.structure.geo import meters_per_pixel
    return meters_per_pixel(16, z16_center_lat)


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class IdRow:
    """One row of the identification-coverage report (v3 only)."""

    gt_id: str
    label: str
    tier: int | None
    candidates: list[str]
    best_v3_anchor_id: str | None
    best_v3_structure_type: str | None
    best_v3_position: str | None
    compass_score: int
    shared_tokens: list[str]
    type_match: bool          # whether v3 structure_type ∈ candidates
    identified: bool          # final binary verdict on identification

    def status(self) -> str:
        if not self.identified:
            return "NOT_FOUND"
        if self.candidates and not self.type_match:
            return "FOUND_TYPE_MISREAD"
        return "FOUND"


@dataclass
class PlacementRow:
    """One row of the placement-accuracy report (coord-gen only)."""

    gt_id: str
    tier: int | None
    v3_anchor_id: str
    gt_pixel_center: tuple[float, float]
    pred_pixel_center: tuple[float, float] | None
    px_dist: float | None
    m_dist: float | None
    placement_confidence: float | None
    out_of_bounds: bool = False  # pred pixel landed outside [0, 1280) — contract violation


@dataclass
class CellResult:
    cell_id: str
    grid_mode: str
    gt_active: list[dict]
    v3_anchors: list[dict]
    coord_anchors: list[dict]
    id_rows: list[IdRow]
    placement_rows: list[PlacementRow]
    v3_pipeline_only_aids: list[str]   # v3 anchor_ids not best-match for any GT
    v3_run_dir: Path
    coord_run_dir: Path
    z16_center: tuple[float, float]
    z16_image_path: Path

    def t12_misses(self) -> list[IdRow]:
        return [r for r in self.id_rows if r.tier in (1, 2) and not r.identified]

    def is_pass(self) -> bool:
        return not self.t12_misses()


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def score_cell(
    cell_id: str, grid_mode: str,
    v3_run_dir: Path, coord_run_dir: Path,
    compass_threshold: int = 2,
) -> CellResult:
    """Two-axis scoring per the user's identification/placement separation.

    `compass_threshold=2` requires at least one cardinal/ordinal token match
    (e.g. both mention "NW"), which keeps generic "center" overlap from
    counting as identification.
    """
    paths = v3_h.CellPaths.for_cell(cell_id)
    gt_payload, gt_active = coord_h._load_gt(paths)
    z16_center = tuple(gt_payload["z16_center"])
    z16_image_path = REPO_ROOT / gt_payload["z16_image"]

    v3_payload, v3_anchor_list = coord_h._load_v3_run(v3_run_dir)
    v3_by_id = {a.get("anchor_id"): a for a in v3_anchor_list if a.get("anchor_id")}

    coord_payload = json.loads((coord_run_dir / "parsed.json").read_text(encoding="utf-8"))
    coord_by_id = {a.get("anchor_id"): a for a in coord_payload.get("anchors", [])
                   if a.get("anchor_id")}

    # ---- AXIS 1: identification coverage (v3 only, no pixel data) ----
    id_rows: list[IdRow] = []
    used_v3_aids: set[str] = set()
    for gt in gt_active:
        gt_text = (gt.get("label", "") + " " + gt.get("description", ""))
        candidates = (gt.get("structure_type_candidates")
                      or gt.get("structure_type_options") or [])
        # Score every v3 anchor, pick the highest. structure_type compatibility
        # acts as a tiebreaker but doesn't gate identification.
        best_aid: str | None = None
        best_score = 0
        best_shared: set[str] = set()
        best_type_match = False
        for aid, v3a in v3_by_id.items():
            v3_text = (v3a.get("position_in_zone", "") + " " + v3a.get("rationale", ""))
            score, shared = _compass_score(gt_text, v3_text)
            type_match = (v3a.get("structure_type") in candidates) if candidates else False
            # Tiebreaker: prefer type-match when scores equal.
            better = (
                score > best_score
                or (score == best_score and type_match and not best_type_match)
            )
            if better:
                best_score = score
                best_aid = aid
                best_shared = shared
                best_type_match = type_match

        identified = best_score >= compass_threshold and best_aid is not None
        if identified:
            used_v3_aids.add(best_aid)
        v3_anchor = v3_by_id.get(best_aid) if best_aid else None
        id_rows.append(IdRow(
            gt_id=gt["gt_id"],
            label=gt.get("label", ""),
            tier=gt.get("tier"),
            candidates=list(candidates),
            best_v3_anchor_id=best_aid if identified else None,
            best_v3_structure_type=v3_anchor.get("structure_type") if v3_anchor and identified else None,
            best_v3_position=v3_anchor.get("position_in_zone") if v3_anchor and identified else None,
            compass_score=best_score,
            shared_tokens=sorted(best_shared),
            type_match=best_type_match if identified else False,
            identified=identified,
        ))

    pipeline_only = [aid for aid in v3_by_id.keys() if aid not in used_v3_aids]

    # ---- AXIS 2: placement accuracy (coord-gen only) ----
    # Contract: pixel coordinates must lie in [0, 1280). Anything outside is a
    # model contract violation, not a placement to compare. We surface those
    # as out_of_bounds and exclude them from distance stats.
    img_size = 1280
    m_per_px = _meters_per_pixel_at(z16_center[0])
    placement_rows: list[PlacementRow] = []
    for row in id_rows:
        if not row.identified or not row.best_v3_anchor_id:
            continue
        cg = coord_by_id.get(row.best_v3_anchor_id)
        gt = next((g for g in gt_active if g["gt_id"] == row.gt_id), None)
        gt_center = tuple(gt["pixel_center"]) if gt else (0.0, 0.0)
        pred_center = None
        px_dist = None
        m_dist = None
        conf = None
        oob = False
        if cg:
            pc = cg.get("pixel_center")
            if pc and len(pc) == 2:
                pred_center = (float(pc[0]), float(pc[1]))
                if not (0 <= pred_center[0] < img_size and 0 <= pred_center[1] < img_size):
                    oob = True  # don't compute distance — placement is invalid
                else:
                    dx = pred_center[0] - gt_center[0]
                    dy = pred_center[1] - gt_center[1]
                    px_dist = (dx * dx + dy * dy) ** 0.5
                    m_dist = px_dist * m_per_px
            conf = cg.get("placement_confidence")
        placement_rows.append(PlacementRow(
            gt_id=row.gt_id,
            tier=row.tier,
            v3_anchor_id=row.best_v3_anchor_id,
            gt_pixel_center=gt_center,
            pred_pixel_center=pred_center,
            px_dist=px_dist,
            m_dist=m_dist,
            placement_confidence=conf,
            out_of_bounds=oob,
        ))

    return CellResult(
        cell_id=cell_id, grid_mode=grid_mode,
        gt_active=gt_active,
        v3_anchors=v3_anchor_list,
        coord_anchors=list(coord_by_id.values()),
        id_rows=id_rows, placement_rows=placement_rows,
        v3_pipeline_only_aids=pipeline_only,
        v3_run_dir=v3_run_dir, coord_run_dir=coord_run_dir,
        z16_center=z16_center, z16_image_path=z16_image_path,
    )


# ------------------------------------------------------------------
# Visual overlay
# ------------------------------------------------------------------


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_id_overlay(result: CellResult) -> Path | None:
    """Render z16 image with GT bboxes (red) + a v3-anchor text legend.

    PURE IDENTIFICATION VIEW — no coord-gen pixel placements. Shows the user
    what v3 thinks the cell contains (in compass-position prose) alongside
    the GT bboxes, so identification quality can be judged without coord-gen
    placement mixed in. The v3 anchor list is rendered as a text legend in
    the bottom-right corner of the image.
    """
    from PIL import Image, ImageDraw

    if not result.z16_image_path.exists():
        return None

    base = Image.open(result.z16_image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(20)
    font_small = _load_font(14)

    GT_COLOR = (220, 30, 30, 255)

    # GT bboxes only — no pipeline data
    for gt in result.gt_active:
        bbox = tuple(gt["pixel_bbox"])
        cx, cy = gt["pixel_center"]
        draw.rectangle(bbox, outline=GT_COLOR, width=4)
        r = 9
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GT_COLOR,
                     outline=(255, 255, 255, 255), width=2)
        tier = gt.get("tier")
        label = f"{gt['gt_id']} T{tier}" if tier is not None else gt["gt_id"]
        tx, ty = bbox[0] + 4, max(bbox[1] - 22, 2)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), label, fill="black", font=font)
        draw.text((tx, ty), label, fill=GT_COLOR, font=font)

    # Text legend in bottom-right: v3 anchor list with compass positions
    legend_lines = [f"v3 anchors ({result.grid_mode}):"]
    for v3a in result.v3_anchors:
        aid = v3a.get("anchor_id", "?")
        stype = (v3a.get("structure_type") or "?")[:18]
        pos = (v3a.get("position_in_zone") or "?")[:55]
        # Mark whether this anchor was best-match for any GT
        matched = aid in {r.best_v3_anchor_id for r in result.id_rows if r.identified}
        prefix = "->" if matched else "  "
        legend_lines.append(f"{prefix} {aid} {stype}: {pos}")

    line_h = 18
    box_w = 560
    box_h = line_h * len(legend_lines) + 14
    box_x0 = base.size[0] - box_w - 8
    box_y0 = base.size[1] - box_h - 8
    draw.rectangle([box_x0, box_y0, box_x0 + box_w, box_y0 + box_h],
                   fill=(0, 0, 0, 200), outline=(255, 255, 255, 255), width=2)
    ty = box_y0 + 6
    for line in legend_lines:
        draw.text((box_x0 + 6, ty), line, fill=(255, 255, 255, 255), font=font_small)
        ty += line_h

    composed = Image.alpha_composite(base, overlay)
    review_dir = result.coord_run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    out_path = review_dir / f"{result.cell_id}_id_overlay.png"
    composed.convert("RGB").save(out_path)
    return out_path


def render_overlay(result: CellResult) -> Path | None:
    """Render z16 image with GT (red) + pipeline (blue) markings.

    PLACEMENT VIEW — shows coord-gen pixel placements vs GT bboxes. Use this
    to judge spatial accuracy. For identification-only review, see
    render_id_overlay() / *_id_overlay.png.
    """
    from PIL import Image, ImageDraw

    if not result.z16_image_path.exists():
        logger.warning("z16 image not found for overlay: %s", result.z16_image_path)
        return None

    base = Image.open(result.z16_image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(20)
    font_small = _load_font(16)

    GT_COLOR = (220, 30, 30, 255)
    GT_FILL = (220, 30, 30, 255)
    PIPE_COLOR = (30, 110, 230, 255)
    PIPE_FILL = (30, 110, 230, 255)
    LINE_COLOR = (140, 60, 200, 200)  # connector

    # GT bboxes + centers (red)
    for gt in result.gt_active:
        bbox = tuple(gt["pixel_bbox"])
        cx, cy = gt["pixel_center"]
        draw.rectangle(bbox, outline=GT_COLOR, width=4)
        r = 9
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GT_FILL,
                     outline=(255, 255, 255, 255), width=2)
        tier = gt.get("tier")
        label = f"{gt['gt_id']} T{tier}" if tier is not None else gt["gt_id"]
        # Place label above bbox; outline for legibility
        tx, ty = bbox[0] + 4, max(bbox[1] - 22, 2)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), label, fill="black", font=font)
        draw.text((tx, ty), label, fill=GT_COLOR, font=font)

    # Pipeline bboxes + centers (blue)
    cg_by_id = {a.get("anchor_id"): a for a in result.coord_anchors}
    v3_by_id = {a.get("anchor_id"): a for a in result.v3_anchors}
    for aid, v3a in v3_by_id.items():
        cg = cg_by_id.get(aid)
        if not cg or not cg.get("pixel_center") or not cg.get("pixel_bbox"):
            continue
        bbox = tuple(int(v) for v in cg["pixel_bbox"])
        cx, cy = float(cg["pixel_center"][0]), float(cg["pixel_center"][1])
        draw.rectangle(bbox, outline=PIPE_COLOR, width=3)
        r = 8
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PIPE_FILL,
                     outline=(255, 255, 255, 255), width=2)
        stype = (v3a.get("structure_type") or "")[:14]
        label = f"{aid} {stype}"
        # Place label below the center
        tx, ty = int(cx) + 12, int(cy) - 8
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), label, fill="black", font=font_small)
        draw.text((tx, ty), label, fill=PIPE_COLOR, font=font_small)

    # Connectors: GT center -> matched pipeline center
    for row in result.id_rows:
        if not row.identified or not row.best_v3_anchor_id:
            continue
        cg = cg_by_id.get(row.best_v3_anchor_id)
        gt = next((g for g in result.gt_active if g["gt_id"] == row.gt_id), None)
        if not cg or not gt or not cg.get("pixel_center"):
            continue
        gx, gy = gt["pixel_center"]
        px, py = float(cg["pixel_center"][0]), float(cg["pixel_center"][1])
        draw.line([(gx, gy), (px, py)], fill=LINE_COLOR, width=2)

    composed = Image.alpha_composite(base, overlay)
    review_dir = result.coord_run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    out_path = review_dir / f"{result.cell_id}_placement_overlay.png"
    composed.convert("RGB").save(out_path)
    return out_path


def write_split_reports(result: CellResult) -> tuple[Path, Path]:
    """Write id_report.json (v3 only) and placement_report.json (coord-gen only).

    Stored alongside the existing combined score_report.json in the coord-gen
    run dir. The split reports are the per-cell artifacts to read when you
    want a clean separation between "did v3 see the structure?" and "did
    coord-gen pin it to the right pixel?". Also colocates the raw v3 LLM
    response and the raw coord-gen response for one-stop per-cell review.
    """
    import shutil
    review_dir = result.coord_run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    # Colocate raw LLM responses + parsed json from BOTH stages so the
    # entire per-cell record is in one folder.
    v3_raw_src = result.v3_run_dir / "raw_response.md"
    v3_parsed_src = result.v3_run_dir / "parsed.json"
    cg_raw_src = result.coord_run_dir / "raw_response.md"
    cg_parsed_src = result.coord_run_dir / "parsed.json"
    if v3_raw_src.exists():
        shutil.copy2(v3_raw_src, review_dir / f"{result.cell_id}_v3_raw_response.md")
    if v3_parsed_src.exists():
        shutil.copy2(v3_parsed_src, review_dir / f"{result.cell_id}_v3_parsed.json")
    if cg_raw_src.exists():
        shutil.copy2(cg_raw_src, review_dir / f"{result.cell_id}_coordgen_raw_response.md")
    if cg_parsed_src.exists():
        shutil.copy2(cg_parsed_src, review_dir / f"{result.cell_id}_coordgen_parsed.json")

    # IDENTIFICATION report — purely v3, no pixel data
    id_payload = {
        "cell_id": result.cell_id,
        "grid_mode": result.grid_mode,
        "v3_run": str(result.v3_run_dir.relative_to(REPO_ROOT)),
        "v3_anchor_list": [
            {
                "anchor_id": a.get("anchor_id"),
                "structure_type": a.get("structure_type"),
                "tier": a.get("tier"),
                "position_in_zone": a.get("position_in_zone"),
                "rationale": (a.get("rationale") or "")[:300],
                "confidence": a.get("confidence"),
                "needs_deeper_zoom": a.get("needs_deeper_zoom"),
            }
            for a in result.v3_anchors
        ],
        "gt_match": [
            {
                "gt_id": r.gt_id,
                "tier": r.tier,
                "label": r.label,
                "candidates": r.candidates,
                "status": r.status(),
                "identified": r.identified,
                "matched_v3_anchor_id": r.best_v3_anchor_id,
                "matched_v3_structure_type": r.best_v3_structure_type,
                "matched_v3_position": r.best_v3_position,
                "compass_score": r.compass_score,
                "shared_tokens": r.shared_tokens,
                "type_match": r.type_match,
            }
            for r in result.id_rows
        ],
        "v3_pipeline_only_anchor_ids": result.v3_pipeline_only_aids,
        "summary": {
            "total_gt": len(result.id_rows),
            "identified": sum(1 for r in result.id_rows if r.identified),
            "type_misreads": sum(1 for r in result.id_rows
                                 if r.identified and r.candidates and not r.type_match),
            "t12_misses": [r.gt_id for r in result.t12_misses()],
        },
    }
    id_path = review_dir / f"{result.cell_id}_id_report.json"
    id_path.write_text(json.dumps(id_payload, indent=2), encoding="utf-8")

    # PLACEMENT report — coord-gen pixel data only, scoped to v3-identified anchors
    placement_payload = {
        "cell_id": result.cell_id,
        "grid_mode": result.grid_mode,
        "coord_run": str(result.coord_run_dir.relative_to(REPO_ROOT)),
        "rows": [
            {
                "gt_id": p.gt_id,
                "tier": p.tier,
                "v3_anchor_id": p.v3_anchor_id,
                "gt_pixel_center": list(p.gt_pixel_center),
                "pred_pixel_center": list(p.pred_pixel_center) if p.pred_pixel_center else None,
                "px_dist": p.px_dist,
                "m_dist": p.m_dist,
                "placement_confidence": p.placement_confidence,
                "out_of_bounds": p.out_of_bounds,
            }
            for p in result.placement_rows
        ],
        "summary": {
            "total_placements": len(result.placement_rows),
            "out_of_bounds": sum(1 for p in result.placement_rows if p.out_of_bounds),
            "valid_distances_m": sorted([p.m_dist for p in result.placement_rows
                                         if p.m_dist is not None]),
        },
    }
    placement_path = review_dir / f"{result.cell_id}_placement_report.json"
    placement_path.write_text(json.dumps(placement_payload, indent=2), encoding="utf-8")
    return id_path, placement_path


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def print_cell_report(result: CellResult) -> None:
    print()
    print("=" * 100)
    verdict = "PASS" if result.is_pass() else "FAIL"
    print(f"CELL {result.cell_id}  (grid_mode={result.grid_mode})  -> {verdict}  "
          f"[{len(result.t12_misses())} T1/T2 misses]")
    print("=" * 100)

    # -- Identification (v3 only) --
    found = sum(1 for r in result.id_rows if r.identified)
    misread = sum(1 for r in result.id_rows
                  if r.identified and r.candidates and not r.type_match)
    print(f"\n[1] IDENTIFICATION  (v3 only — compass-token overlap)")
    print(f"    found {found}/{len(result.id_rows)} GT anchors  "
          f"({misread} with feature-type misread)")
    print(f"    {'gt':<5s} {'tier':<4s} {'status':<19s} {'aid':<5s} {'score':>5s}  "
          f"{'shared':<22s} v3_position / label")
    for r in result.id_rows:
        tier = f"T{r.tier}" if r.tier is not None else "T?"
        aid = r.best_v3_anchor_id or "-"
        shared = ",".join(r.shared_tokens) if r.shared_tokens else "-"
        line = f"    {r.gt_id:<5s} {tier:<4s} {r.status():<19s} {aid:<5s} {r.compass_score:>5d}  {shared[:22]:<22s} "
        if r.identified:
            line += f"{(r.best_v3_position or '')[:50]}"
        else:
            line += f"GT: {r.label[:50]}"
        print(line)
    if result.v3_pipeline_only_aids:
        print(f"\n    v3 anchors not best-match for any GT (potential additional coverage):")
        for aid in result.v3_pipeline_only_aids:
            v3a = next((a for a in result.v3_anchors if a.get("anchor_id") == aid), None)
            if v3a:
                print(f"      {aid:<5s} {v3a.get('structure_type', '?'):<22s} "
                      f"pos='{(v3a.get('position_in_zone') or '')[:50]}'")

    # -- Placement (coord-gen only, given v3 found something) --
    oob_count = sum(1 for p in result.placement_rows if p.out_of_bounds)
    print(f"\n[2] PLACEMENT  (coord-gen only — distance from predicted to GT center)")
    if oob_count:
        print(f"    {oob_count} placement(s) INVALID — pixel center outside [0, 1280)")
    if not result.placement_rows:
        print("    (no v3-identified anchors to evaluate)")
    else:
        print(f"    {'gt':<5s} {'tier':<4s} {'aid':<5s} {'pred_px':<14s} "
              f"{'gt_px':<14s} {'px_dist':>8s} {'m_dist':>8s}  conf  status")
        m_dists = []
        for p in result.placement_rows:
            tier = f"T{p.tier}" if p.tier is not None else "T?"
            pred = f"({p.pred_pixel_center[0]:.0f},{p.pred_pixel_center[1]:.0f})" \
                if p.pred_pixel_center else "-"
            gt_c = f"({p.gt_pixel_center[0]:.0f},{p.gt_pixel_center[1]:.0f})"
            pxd = f"{p.px_dist:.1f}" if p.px_dist is not None else "-"
            md = f"{p.m_dist:.1f}" if p.m_dist is not None else "-"
            cf = f"{p.placement_confidence:.2f}" if p.placement_confidence is not None else "?"
            status = "INVALID" if p.out_of_bounds else "ok"
            print(f"    {p.gt_id:<5s} {tier:<4s} {p.v3_anchor_id:<5s} {pred:<14s} "
                  f"{gt_c:<14s} {pxd:>8s} {md:>8s}  {cf}  {status}")
            if p.m_dist is not None:
                m_dists.append(p.m_dist)
        if m_dists:
            srt = sorted(m_dists)
            med = srt[len(srt) // 2]
            print(f"    placement error (m, valid only): min={min(m_dists):.1f}  "
                  f"med={med:.1f}  max={max(m_dists):.1f}  n={len(m_dists)}")
    print("=" * 100)


def print_overall_summary(results: list[CellResult]) -> None:
    print()
    print("#" * 100)
    print("OVERALL SUMMARY")
    print("#" * 100)
    # Identification totals by tier
    tally = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], None: [0, 0, 0]}  # [found, type_misread, total]
    for r in results:
        for row in r.id_rows:
            t = row.tier if row.tier in (1, 2, 3) else None
            tally[t][2] += 1
            if row.identified:
                tally[t][0] += 1
                if row.candidates and not row.type_match:
                    tally[t][1] += 1
    print(f"\nIDENTIFICATION (compass-token, threshold >= 2):")
    for t_label, key in (("T1", 1), ("T2", 2), ("T3", 3), ("T?", None)):
        found, misread, total = tally[key]
        if total == 0:
            continue
        pct = (100.0 * found / total) if total else 0
        print(f"  {t_label}: found {found}/{total} ({pct:.0f}%)  "
              f"of which {misread} with feature-type misread")
    # Placement aggregates
    all_dists = [p.m_dist for r in results for p in r.placement_rows if p.m_dist is not None]
    oob_total = sum(p.out_of_bounds for r in results for p in r.placement_rows)
    placement_total = sum(len(r.placement_rows) for r in results)
    print(f"\nPLACEMENT (coord-gen):")
    print(f"  total placements scored: {placement_total}  ({oob_total} INVALID — out of bounds)")
    if all_dists:
        srt = sorted(all_dists)
        med = srt[len(srt) // 2]
        print(f"  valid distances (m): n={len(all_dists)}  min={min(all_dists):.1f}  "
              f"med={med:.1f}  max={max(all_dists):.1f}  mean={sum(all_dists)/len(all_dists):.1f}")
    else:
        print("  no valid placement rows.")
    # Pass/fail
    failed = [r.cell_id for r in results if not r.is_pass()]
    print()
    if failed:
        print(f"FAILED (T1/T2 unidentified): {failed}")
    else:
        print("ALL CELLS PASS (every T1/T2 GT was identified by v3).")
    print("#" * 100)


# ------------------------------------------------------------------
# Sweep orchestration
# ------------------------------------------------------------------


async def run_one_cell(
    cell_id: str, file_mode: str, replay: bool,
    inject_evidence: bool = False, coords_mode: str = "grid",
) -> tuple[Path | None, Path | None]:
    """Run v3 + coord-gen for one cell. Returns (v3_run_dir, coord_run_dir)."""
    paths = v3_h.CellPaths.for_cell(cell_id)
    if replay:
        v3_run = v3_h._find_latest_run(file_mode, paths)
        coord_run = coord_h._find_latest_coord_run(cell_id)
        if v3_run is None:
            logger.error("no v3_%s_%s run on disk for replay", cell_id, file_mode)
            return None, None
        if coord_run is None:
            logger.error("no coord_%s run on disk for replay", cell_id)
            return v3_run, None
        return v3_run, coord_run

    # v3 first
    parsed_v3, raw_v3 = await v3_h.call_anchor_identification(
        file_mode, paths, inject_evidence=inject_evidence,
    )
    anchors_v3 = parsed_v3.get("anchors") if isinstance(parsed_v3, dict) else []
    if not isinstance(anchors_v3, list):
        anchors_v3 = []
    v3_run = v3_h.save_run(file_mode, parsed_v3, raw_v3, [], list(anchors_v3), paths)
    logger.info("[%s] v3 anchors: %d  -> %s", cell_id, len(anchors_v3),
                v3_run.relative_to(REPO_ROOT))

    if not anchors_v3:
        logger.warning("[%s] v3 returned no anchors; skipping coord-gen", cell_id)
        return v3_run, None

    # coord-gen against the v3 anchors we just produced
    gt_payload, _ = coord_h._load_gt(paths)
    z16_center = tuple(gt_payload["z16_center"])
    try:
        parsed_cg, raw_cg = await coord_h.call_coord_gen(
            paths, z16_center, anchors_v3, coords_mode=coords_mode,
        )
    except RuntimeError as exc:
        logger.error("[%s] coord-gen failed: %s", cell_id, exc)
        return v3_run, None
    coord_anchors = parsed_cg.get("anchors") or []
    rows, summary = coord_h.score_run(anchors_v3, coord_anchors, z16_center,
                                      [a for a in gt_payload["anchors"]
                                       if a.get("status") == "active"])
    coord_run = coord_h.save_run(cell_id, v3_run, parsed_cg, raw_cg, rows, summary)
    logger.info("[%s] coord-gen placements: %d  -> %s",
                cell_id, len(coord_anchors), coord_run.relative_to(REPO_ROOT))
    return v3_run, coord_run


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-mode", choices=("nogrid", "grid"), default="nogrid",
                        help="v3 prompt variant to run (default: nogrid)")
    parser.add_argument("--coords-mode", choices=("nogrid", "grid"), default="grid",
                        help="coord-gen prompt variant (default: grid)")
    parser.add_argument("--evidence", action="store_true",
                        help="Inject NAIP/NOAA/FWC habitat evidence into the v3 prompt")
    parser.add_argument("--replay", action="store_true",
                        help="Use the latest v3+coord-gen runs already on disk")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Limit to specific cells")
    args = parser.parse_args()

    cells = _discover_cells_with_gt()
    if args.only:
        wanted = set(args.only)
        cells = [c for c in cells if c in wanted]
    if not cells:
        print("no cells with gt_anchors.json found.", file=sys.stderr)
        return 2

    if not args.replay and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.", file=sys.stderr)
        return 2

    print(f"sweep: {len(cells)} cell(s), grid_mode={args.grid_mode}, "
          f"coords_mode={args.coords_mode}, evidence={args.evidence}, replay={args.replay}")
    results: list[CellResult] = []
    for cid in cells:
        v3_run, coord_run = await run_one_cell(
            cid, args.grid_mode, args.replay,
            inject_evidence=args.evidence, coords_mode=args.coords_mode,
        )
        if coord_run is None:
            logger.warning("skipping %s in coverage report (no coord run)", cid)
            continue
        result = score_cell(cid, args.grid_mode, v3_run, coord_run)
        print_cell_report(result)
        id_overlay = render_id_overlay(result)
        placement_overlay = render_overlay(result)
        id_report_path, placement_report_path = write_split_reports(result)
        print(f"    id-overlay       : {id_overlay.relative_to(REPO_ROOT) if id_overlay else '-'}")
        print(f"    placement-overlay: {placement_overlay.relative_to(REPO_ROOT) if placement_overlay else '-'}")
        print(f"    id-report        : {id_report_path.relative_to(REPO_ROOT)}")
        print(f"    placement-report : {placement_report_path.relative_to(REPO_ROOT)}")
        results.append(result)

    print_overall_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
