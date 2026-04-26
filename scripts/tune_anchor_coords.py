"""Iteration harness for tuning the coord-gen prompts (TASK-2).

Takes a v3 anchor-identification run (the parsed.json from
`tune_anchor_identification_v3.py`), sends the anchor list + the
grid-overlaid z16 image to the coord-gen prompt pair, and scores the
per-anchor placement against the hand-labeled ground truth at
`data/areas/rookery_bay_v2/images/structures/root-10-8/gt_anchors.json`.

Implements the addendum-locked policies from `docs/PHASE_C_TASKS.md`:

  - Matching is STRICT by `anchor_id`. There is no fuzzy "structure_type +
    position overlap" fallback — if coord-gen returns an `anchor_id` that
    is not in the v3 input list (or fails to return one), the harness
    increments an `unmatched_anchors` counter in the report and moves on.
  - Per-anchor failures (out-of-bounds pixel, very low placement_confidence)
    are reported as `Finding`-style flags. The anchor stays in the output.
  - A wholly malformed batch JSON (no `anchors` list, JSON parse error)
    exits the harness with a non-zero status.

Usage:
  python scripts/tune_anchor_coords.py
  python scripts/tune_anchor_coords.py --v3-run data/areas/rookery_bay_v2/images/tuning_runs/v3_grid_20260424-...
  python scripts/tune_anchor_coords.py --replay
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
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

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from readwater.api.claude_vision import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    _extract_json_from_response,
    _get_client,
    _load_prompt,
)
from readwater.pipeline.structure.geo import pixel_to_latlon  # noqa: E402

# Import the v3 harness so we can reuse Z16_IMAGE / Z16_GRID_OVERLAY paths
# and the grid-overlay rendering helper. Keeps the two harnesses in sync on
# tile geometry.
import tune_anchor_identification_v3 as v3harness  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_anchor_coords")


# ------------------------------------------------------------------
# Paths + constants
# ------------------------------------------------------------------

DEFAULT_CELL_ID = v3harness.DEFAULT_CELL_ID
IMG_SIZE_PX = 1280
ZOOM = 16

V3_RUNS_DIR = v3harness.RUNS_DIR
COORD_RUNS_DIR = (
    REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images" / "coord_runs"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_block(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _b64_png(path),
        },
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU for axis-aligned bboxes [x0, y0, x1, y1]. 0 if disjoint or invalid."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pixel_in_bounds(px: float, py: float) -> bool:
    return 0 <= px < IMG_SIZE_PX and 0 <= py < IMG_SIZE_PX


# ------------------------------------------------------------------
# Run discovery + loading
# ------------------------------------------------------------------


def _find_latest_v3_run(cell_id: str) -> Path | None:
    if not V3_RUNS_DIR.exists():
        return None
    # Cell-tagged dirs first (post multi-cell refactor)
    candidates = sorted(V3_RUNS_DIR.glob(f"v3_{cell_id}_*"), reverse=True)
    if candidates:
        return candidates[0]
    # Legacy untagged for the default cell only
    if cell_id == DEFAULT_CELL_ID:
        legacy = sorted(V3_RUNS_DIR.glob("v3_*"), reverse=True)
        legacy = [c for c in legacy if not c.name.startswith(f"v3_{cell_id}_")]
        if legacy:
            return legacy[0]
    return None


def _find_latest_coord_run(cell_id: str) -> Path | None:
    if not COORD_RUNS_DIR.exists():
        return None
    candidates = sorted(COORD_RUNS_DIR.glob(f"coord_{cell_id}_*"), reverse=True)
    if candidates:
        return candidates[0]
    if cell_id == DEFAULT_CELL_ID:
        # Legacy untagged
        legacy = sorted(COORD_RUNS_DIR.glob("coord_*"), reverse=True)
        legacy = [c for c in legacy if not c.name.startswith(f"coord_{cell_id}_")]
        if legacy:
            return legacy[0]
    return None


def _load_v3_run(run_dir: Path) -> tuple[dict, list[dict]]:
    parsed_path = run_dir / "parsed.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"v3 run dir is missing parsed.json: {run_dir}")
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    anchors = parsed.get("anchors") or []
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"v3 run has no usable anchors[]: {run_dir}")
    return parsed, anchors


def _load_gt(paths: v3harness.CellPaths) -> tuple[dict, list[dict]]:
    if not paths.gt_anchors_json.exists():
        raise FileNotFoundError(f"GT file not found: {paths.gt_anchors_json}")
    payload = json.loads(paths.gt_anchors_json.read_text(encoding="utf-8"))
    return payload, [a for a in payload["anchors"] if a.get("status") == "active"]


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------


COORDS_FILE_MODES = ("nogrid", "grid")


async def call_coord_gen(
    paths: v3harness.CellPaths,
    z16_center: tuple[float, float],
    anchors_in: list[dict],
    coords_mode: str = "grid",
) -> tuple[dict, str]:
    """Run the coord-gen prompt pair. Returns (parsed_json, raw_text).

    coords_mode is "nogrid" (clean image, compass-only reasoning) or
    "grid" (8x8 A1-H8 overlay drawn on the image).
    """
    if coords_mode not in COORDS_FILE_MODES:
        raise ValueError(f"coords_mode must be one of {COORDS_FILE_MODES}, got {coords_mode!r}")
    system_prompt = _load_prompt(f"anchor_coords_{coords_mode}_system.txt")
    user_template = _load_prompt(f"anchor_coords_{coords_mode}_user.txt")

    if coords_mode == "grid":
        coord_image = v3harness._ensure_grid_overlay(paths)
        image_label = "IMAGE 1 — z16_local with 8x8 A1-H8 grid overlay (1280x1280):"
    else:
        coord_image = paths.z16_image
        image_label = "IMAGE 1 — z16_local clean image (1280x1280, no overlay):"
    cell_id = paths.cell_id

    user_prompt = user_template.format(
        cell_id=cell_id,
        z16_center_lat=f"{z16_center[0]:.6f}",
        z16_center_lon=f"{z16_center[1]:.6f}",
        anchor_list_json=json.dumps(anchors_in, indent=2),
    )

    content = [
        {"type": "text", "text": image_label},
        _image_block(coord_image),
        {"type": "text", "text": user_prompt},
    ]

    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw_text = response.content[0].text
    try:
        parsed = _extract_json_from_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        # Per addendum: malformed batch JSON is a stage-failure (no
        # recovery possible). Bubble up with a useful message.
        raise RuntimeError(f"coord-gen returned unparseable JSON: {exc}") from exc
    return parsed, raw_text


# ------------------------------------------------------------------
# Match + score
# ------------------------------------------------------------------


@dataclass
class CoordRow:
    """One row of the per-anchor scoring report."""

    anchor_id: str
    structure_type: str | None
    pixel_center: tuple[float, float] | None
    pixel_bbox: tuple[int, int, int, int] | None
    pred_latlon: tuple[float, float] | None
    placement_confidence: float | None
    placement_notes: str
    matched_gt_id: str | None
    gt_label: str | None
    gt_latlon: tuple[float, float] | None
    gt_pixel_bbox: tuple[int, int, int, int] | None
    meters_error: float | None
    bbox_iou: float | None
    findings: list[dict]


def score_run(
    v3_anchors: list[dict],
    coord_anchors: list[dict],
    z16_center: tuple[float, float],
    gt_anchors: list[dict],
) -> tuple[list[CoordRow], dict]:
    """Strict-by-anchor_id matching. Returns (rows, summary)."""
    coord_by_id = {a.get("anchor_id"): a for a in coord_anchors if a.get("anchor_id")}
    gt_by_label_token: dict[str, dict] = {}
    # GT anchors don't carry v3 anchor_ids — they are labeled by gt_id and
    # rich descriptions. Match coord-gen output to GT by nearest haversine
    # distance using the prose latlon (the v3 anchor a coord-gen entry
    # describes is itself looked up by anchor_id; the GT lookup is then
    # geometric, since GT is the ground truth for any anchor pinned in this
    # cell regardless of which v3 surface generated it).
    # This is NOT the contradicted "fuzzy fallback" — this is the GT lookup,
    # which has no anchor_id concept on the GT side.
    rows: list[CoordRow] = []
    unmatched_v3 = 0
    unmatched_coord = 0

    v3_id_set = {a.get("anchor_id") for a in v3_anchors}

    # Surface coord_gen entries that refer to anchor_ids the v3 run never
    # produced (a contract violation, not a placement quality issue).
    for cg_id in coord_by_id.keys():
        if cg_id not in v3_id_set:
            unmatched_coord += 1
            logger.warning(
                "coord-gen returned anchor_id %r not present in v3 input — ignored",
                cg_id,
            )

    for v3a in v3_anchors:
        a_id = v3a.get("anchor_id")
        cg = coord_by_id.get(a_id)
        if cg is None:
            unmatched_v3 += 1
            rows.append(CoordRow(
                anchor_id=a_id,
                structure_type=v3a.get("structure_type"),
                pixel_center=None,
                pixel_bbox=None,
                pred_latlon=None,
                placement_confidence=None,
                placement_notes="",
                matched_gt_id=None,
                gt_label=None,
                gt_latlon=None,
                gt_pixel_bbox=None,
                meters_error=None,
                bbox_iou=None,
                findings=[{
                    "issue_code": "NO_COORD_RESPONSE",
                    "severity": "warn",
                    "message": "coord-gen returned no entry for this anchor_id",
                }],
            ))
            continue

        # Pull placement
        try:
            pc = cg.get("pixel_center") or [None, None]
            pb = cg.get("pixel_bbox") or [None, None, None, None]
            cx = float(pc[0]) if pc[0] is not None else None
            cy = float(pc[1]) if pc[1] is not None else None
            bbox = (int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])) \
                if all(v is not None for v in pb) else None
            conf = float(cg.get("placement_confidence", 0.0))
        except (ValueError, TypeError) as exc:
            rows.append(CoordRow(
                anchor_id=a_id,
                structure_type=v3a.get("structure_type"),
                pixel_center=None, pixel_bbox=None, pred_latlon=None,
                placement_confidence=None,
                placement_notes=str(cg.get("placement_notes", "")),
                matched_gt_id=None, gt_label=None, gt_latlon=None,
                gt_pixel_bbox=None, meters_error=None, bbox_iou=None,
                findings=[{
                    "issue_code": "MALFORMED_PLACEMENT",
                    "severity": "warn",
                    "message": f"could not parse placement fields: {exc}",
                }],
            ))
            continue

        findings: list[dict] = []
        if cx is None or cy is None:
            findings.append({
                "issue_code": "MISSING_PIXEL_CENTER",
                "severity": "warn",
                "message": "pixel_center missing",
            })
            pred_latlon = None
        else:
            if not _pixel_in_bounds(cx, cy):
                findings.append({
                    "issue_code": "COORDS_OUT_OF_BOUNDS",
                    "severity": "warn",
                    "message": f"pixel_center ({cx},{cy}) outside [0,{IMG_SIZE_PX})",
                })
                # Clamp before converting so downstream report still works.
                cx = max(0.0, min(float(IMG_SIZE_PX - 1), cx))
                cy = max(0.0, min(float(IMG_SIZE_PX - 1), cy))
            pred_latlon = pixel_to_latlon(
                cx, cy, IMG_SIZE_PX, z16_center[0], z16_center[1], ZOOM,
            )

        if conf < 0.3:
            findings.append({
                "issue_code": "LOW_CONFIDENCE",
                "severity": "info",
                "message": f"placement_confidence={conf:.2f} (<0.30)",
            })

        # GT lookup — nearest active GT entry by haversine, no anchor_id.
        # This is OK because GT is keyed by gt_id, not v3 anchor_id; two
        # different prompts can place the same world feature with different
        # anchor_ids and both should be scored against the same GT.
        matched_gt_id = None
        gt_label = None
        gt_latlon = None
        gt_pixel_bbox = None
        meters_error = None
        bbox_iou = None
        if pred_latlon is not None and gt_anchors:
            # Choose the nearest GT by haversine.
            best = min(
                gt_anchors,
                key=lambda gt: _haversine_m(
                    pred_latlon[0], pred_latlon[1],
                    gt["latlon_center"][0], gt["latlon_center"][1],
                ),
            )
            matched_gt_id = best["gt_id"]
            gt_label = best["label"]
            gt_latlon = tuple(best["latlon_center"])
            gt_pixel_bbox = tuple(best["pixel_bbox"])
            meters_error = _haversine_m(
                pred_latlon[0], pred_latlon[1],
                gt_latlon[0], gt_latlon[1],
            )
            if bbox is not None:
                bbox_iou = _bbox_iou(bbox, gt_pixel_bbox)

        rows.append(CoordRow(
            anchor_id=a_id,
            structure_type=v3a.get("structure_type"),
            pixel_center=(cx, cy) if cx is not None else None,
            pixel_bbox=bbox,
            pred_latlon=pred_latlon,
            placement_confidence=conf,
            placement_notes=str(cg.get("placement_notes", "")),
            matched_gt_id=matched_gt_id,
            gt_label=gt_label,
            gt_latlon=gt_latlon,
            gt_pixel_bbox=gt_pixel_bbox,
            meters_error=meters_error,
            bbox_iou=bbox_iou,
            findings=findings,
        ))

    summary = {
        "v3_anchor_count": len(v3_anchors),
        "coord_anchor_count": len(coord_anchors),
        "unmatched_v3_anchors": unmatched_v3,
        "unmatched_coord_anchors": unmatched_coord,
        "rows_scored": len([r for r in rows if r.meters_error is not None]),
    }
    return rows, summary


def print_report(rows: list[CoordRow], summary: dict, cell_id: str) -> None:
    print()
    print("=" * 96)
    print(f"COORD-GEN TUNING REPORT — {cell_id}")
    print("=" * 96)
    print(f"v3 anchors in : {summary['v3_anchor_count']}")
    print(f"coord-gen out : {summary['coord_anchor_count']}")
    print(f"unmatched v3  : {summary['unmatched_v3_anchors']} "
          "(v3 anchor with no coord-gen response)")
    print(f"unmatched cg  : {summary['unmatched_coord_anchors']} "
          "(coord-gen returned an anchor_id v3 didn't emit)")
    print(f"rows scored   : {summary['rows_scored']}")
    print()
    print(f"{'aid':<5s}  {'gt':<5s}  {'type':<18s}  {'conf':>4s}  "
          f"{'err_m':>7s}  {'iou':>4s}  flags")
    print("-" * 96)
    err_values: list[float] = []
    iou_values: list[float] = []
    for r in rows:
        flags = ",".join(f["issue_code"] for f in r.findings) or "-"
        err = f"{r.meters_error:7.1f}" if r.meters_error is not None else "    n/a"
        iou = f"{r.bbox_iou:4.2f}" if r.bbox_iou is not None else " n/a"
        conf = f"{r.placement_confidence:4.2f}" if r.placement_confidence is not None else " n/a"
        st = (r.structure_type or "")[:18]
        gtid = r.matched_gt_id or "-"
        print(f"{r.anchor_id or '-':<5s}  {gtid:<5s}  {st:<18s}  {conf}  "
              f"{err}  {iou}  {flags}")
        if r.meters_error is not None:
            err_values.append(r.meters_error)
        if r.bbox_iou is not None:
            iou_values.append(r.bbox_iou)
    print()
    if err_values:
        print(f"meters error     : min={min(err_values):.1f}  "
              f"med={sorted(err_values)[len(err_values)//2]:.1f}  "
              f"max={max(err_values):.1f}  "
              f"mean={sum(err_values)/len(err_values):.1f}")
    if iou_values:
        print(f"bbox IoU         : min={min(iou_values):.2f}  "
              f"med={sorted(iou_values)[len(iou_values)//2]:.2f}  "
              f"max={max(iou_values):.2f}  "
              f"mean={sum(iou_values)/len(iou_values):.2f}")
    print("=" * 96)


def save_run(
    cell_id: str,
    v3_run_dir: Path,
    parsed: dict,
    raw_text: str,
    rows: list[CoordRow],
    summary: dict,
) -> Path:
    COORD_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = COORD_RUNS_DIR / f"coord_{cell_id}_{ts}"
    out.mkdir()
    (out / "raw_response.md").write_text(raw_text, encoding="utf-8")
    (out / "parsed.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    (out / "score_report.json").write_text(json.dumps({
        "cell_id": cell_id,
        "v3_run": str(v3_run_dir.relative_to(REPO_ROOT)),
        "summary": summary,
        "rows": [
            {
                "anchor_id": r.anchor_id,
                "structure_type": r.structure_type,
                "pixel_center": list(r.pixel_center) if r.pixel_center else None,
                "pixel_bbox": list(r.pixel_bbox) if r.pixel_bbox else None,
                "pred_latlon": list(r.pred_latlon) if r.pred_latlon else None,
                "placement_confidence": r.placement_confidence,
                "placement_notes": r.placement_notes,
                "matched_gt_id": r.matched_gt_id,
                "gt_label": r.gt_label,
                "gt_latlon": list(r.gt_latlon) if r.gt_latlon else None,
                "gt_pixel_bbox": list(r.gt_pixel_bbox) if r.gt_pixel_bbox else None,
                "meters_error": r.meters_error,
                "bbox_iou": r.bbox_iou,
                "findings": r.findings,
            }
            for r in rows
        ],
    }, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell", type=str, default=DEFAULT_CELL_ID,
        help=f"Which cell to score (default {DEFAULT_CELL_ID})",
    )
    parser.add_argument(
        "--coords-mode", choices=COORDS_FILE_MODES, default="grid",
        help="Coord-gen prompt variant: 'nogrid' (clean image) or 'grid' "
             "(8x8 A1-H8 overlay)",
    )
    parser.add_argument(
        "--v3-run", type=str, default=None,
        help="Path to a v3 tuning run dir (defaults to the most recent v3_* "
             "for the chosen --cell under data/areas/.../tuning_runs/)",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Re-score the most recent coord_* run on disk for this cell "
             "without calling the API",
    )
    args = parser.parse_args()

    cell_paths = v3harness.CellPaths.for_cell(args.cell)
    gt_payload, gt_active = _load_gt(cell_paths)
    z16_center = tuple(gt_payload["z16_center"])

    if args.replay:
        coord_dir = _find_latest_coord_run(cell_paths.cell_id)
        if coord_dir is None:
            print(f"no coord_{cell_paths.cell_id}_* runs on disk to replay")
            return 2
        replay_payload = json.loads(
            (coord_dir / "score_report.json").read_text(encoding="utf-8"),
        )
        v3_run_dir = REPO_ROOT / replay_payload["v3_run"]
        parsed = json.loads((coord_dir / "parsed.json").read_text(encoding="utf-8"))
        raw_text = (coord_dir / "raw_response.md").read_text(encoding="utf-8")
        logger.info("replaying %s (over v3 %s)", coord_dir.name, v3_run_dir.name)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set; aborting.")
            return 2
        if args.v3_run:
            v3_run_dir = Path(args.v3_run)
            if not v3_run_dir.is_absolute():
                v3_run_dir = REPO_ROOT / v3_run_dir
        else:
            v3_run_dir = _find_latest_v3_run(cell_paths.cell_id)
            if v3_run_dir is None:
                print(f"no v3_{cell_paths.cell_id}_* runs on disk; run "
                      f"`tune_anchor_identification_v3.py --cell {cell_paths.cell_id}` first.")
                return 2
        logger.info("using v3 run: %s", v3_run_dir.relative_to(REPO_ROOT))

    _v3_parsed, v3_anchors = _load_v3_run(v3_run_dir)

    if args.replay:
        coord_anchors = parsed.get("anchors") or []
    else:
        try:
            parsed, raw_text = await call_coord_gen(
                cell_paths, z16_center, v3_anchors, coords_mode=args.coords_mode,
            )
        except RuntimeError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        coord_anchors = parsed.get("anchors") or []
        if not isinstance(coord_anchors, list):
            print(f"FAIL: coord-gen response has no anchors[] list: {parsed!r}",
                  file=sys.stderr)
            return 2

    rows, summary = score_run(v3_anchors, coord_anchors, z16_center, gt_active)
    print_report(rows, summary, cell_paths.cell_id)
    if not args.replay:
        out_dir = save_run(cell_paths.cell_id, v3_run_dir, parsed, raw_text, rows, summary)
        print(f"\nartifacts saved to: {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
