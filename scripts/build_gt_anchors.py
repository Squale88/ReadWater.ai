"""Build pixel-precise gt_anchors.json files for every cell with hand-labeled GT.

Generalizes the original `build_gt_anchors_root_10_8.py` to handle every
cell that has an annotated PNG + prose .txt pair under
`ground_truth/anchors/`. Per the GT-usage policy in
`ground_truth/anchors/Purpose of ground truth.txt`:

  - Tier 1 + Tier 2 anchors are authoritative for coverage.
  - Tier 3 misses are noted, not failures.
  - Pipeline-only anchors are positive signals, not false positives.

This script extracts a `tier` for every GT anchor from the prose body so
downstream scoring can weight Tier 1/2 misses harder than Tier 3.

Per-cell overrides (e.g. the addendum-locked statuses on root-10-8) live
in `_PER_CELL_OVERRIDES` below. Cells with no override use auto-derived
metadata: tier from prose body, candidate structure types from a keyword
sweep on the body, status defaulted to "active".

Usage:
  python scripts/build_gt_anchors.py                  # all cells with GT
  python scripts/build_gt_anchors.py --cell root-2-9
  python scripts/build_gt_anchors.py --check-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
WORKTREE_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(WORKTREE_ROOT / "src"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts"))

from _cells import _sub_cell_center  # noqa: E402
from readwater.pipeline.structure.geo import (  # noqa: E402
    latlon_to_pixel,
    pixel_to_latlon,
)

GT_DIR = REPO_ROOT / "ground_truth" / "anchors"
DATA_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"

ZOOM = 16
IMG_SIZE_PX = 1280


# ------------------------------------------------------------------
# Per-cell overrides — addendum-locked policies for root-10-8 (TASK-0)
# ------------------------------------------------------------------

# Each override entry is {anchor_num (int from prose) -> dict with
# any of: structure_type_options, status, expected_needs_deeper_zoom,
# tier_override, review_note}.
_PER_CELL_OVERRIDES: dict[str, dict[int, dict]] = {
    "root-10-8": {
        1: {"structure_type_options": ["drain_system", "creek_mouth_system"],
            "tier_override": 1, "status": "active", "expected_needs_deeper_zoom": False},
        2: {"structure_type_options": ["island"], "tier_override": 1,
            "status": "active", "expected_needs_deeper_zoom": False},
        3: {"structure_type_options": ["island"], "tier_override": 1,
            "status": "active", "expected_needs_deeper_zoom": False},
        4: {"structure_type_options": ["island"], "tier_override": 1,
            "status": "active", "expected_needs_deeper_zoom": False},
        5: {"structure_type_options": ["point"], "tier_override": 3,
            "status": "under_review",
            "expected_needs_deeper_zoom": False,
            "review_note": "Classification debatable; user research pending."},
        6: {"structure_type_options": ["trough", "current_split"],
            "tier_override": 3, "status": "active",
            "expected_needs_deeper_zoom": False},
        7: {"structure_type_options": ["seagrass_patch", "sand_lobe", "shallow_flat"],
            "tier_override": None, "status": "active",
            "expected_needs_deeper_zoom": True,
            "review_note": "Ambiguous bottom type; v3 should mark needs_deeper_zoom."},
        8: {"structure_type_options": ["island"], "tier_override": 1,
            "status": "active", "expected_needs_deeper_zoom": False},
    },
}


# ------------------------------------------------------------------
# Cell discovery
# ------------------------------------------------------------------


@dataclass
class GTCellSpec:
    cell_id: str          # e.g. "root-10-8"
    parent_id: str        # e.g. "root-10"
    cell_num: int         # e.g. 8
    annotated_png: Path
    prose_txt: Path
    z16_source_image: Path
    out_dir: Path


_GT_FILE_RE = re.compile(r"^z0_(\d+)_(\d+)_anchors\.txt$")


def discover_gt_cells() -> list[GTCellSpec]:
    """Find every (.png, .txt) pair under ground_truth/anchors/ that names
    a z0_<parentnum>_<cellnum> sub-cell."""
    cells: list[GTCellSpec] = []
    for txt in sorted(GT_DIR.glob("z0_*_anchors.txt")):
        m = _GT_FILE_RE.match(txt.name)
        if not m:
            continue
        parent_num = int(m.group(1))
        cell_num = int(m.group(2))
        parent_id = f"root-{parent_num}"
        cell_id = f"{parent_id}-{cell_num}"
        png = txt.with_suffix(".png")
        if not png.exists():
            print(f"  skip {cell_id}: no matching PNG at {png}")
            continue
        z16_source = DATA_DIR / f"z0_{parent_num}_{cell_num}.png"
        out_dir = DATA_DIR / "structures" / cell_id
        cells.append(GTCellSpec(
            cell_id=cell_id, parent_id=parent_id, cell_num=cell_num,
            annotated_png=png, prose_txt=txt,
            z16_source_image=z16_source, out_dir=out_dir,
        ))
    return cells


# ------------------------------------------------------------------
# Prose parsing (with tier extraction)
# ------------------------------------------------------------------


@dataclass
class ProseAnchor:
    num: int
    label_prose: str
    lat: float
    lon: float
    description: str
    tier: int | None
    tier_note: str  # populated when prose hedges (e.g. "Tier 2/3", "Tier 1 if detached; Tier 2 if connected")


_HEADER_RE = re.compile(
    r"^\s*(?:Anchor|Box)\s+(\d+)\s*[—\-]\s*(.+?)\s+(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$"
)
_TIER_RE = re.compile(r"\bTier\s*([0-9])\b", re.IGNORECASE)
_TIER_HEDGE_RE = re.compile(r"\bTier\s*[0-9]\s*/\s*[0-9]\b", re.IGNORECASE)


def _extract_tier(body: str) -> tuple[int | None, str]:
    """Find the primary tier in a body paragraph. Return (tier, hedge_note)."""
    matches = _TIER_RE.findall(body)
    hedge_match = _TIER_HEDGE_RE.search(body)
    if not matches:
        return None, ""
    primary = int(matches[0])
    if hedge_match:
        return primary, hedge_match.group(0)
    if len(matches) > 1 and len(set(matches)) > 1:
        # Multiple distinct tiers mentioned (e.g. "Tier 1 if detached; Tier 2 if connected")
        return primary, "; ".join(f"Tier {t}" for t in matches[:3])
    return primary, ""


def parse_prose(text: str) -> list[ProseAnchor]:
    out: list[ProseAnchor] = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        m = _HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        num = int(m.group(1))
        label = m.group(2).strip()
        lat = float(m.group(3))
        lon = float(m.group(4))
        body = lines[i + 1].strip() if i + 1 < len(lines) else ""
        tier, tier_note = _extract_tier(body)
        out.append(ProseAnchor(
            num=num, label_prose=label, lat=lat, lon=lon,
            description=body, tier=tier, tier_note=tier_note,
        ))
        i += 2
    return out


# ------------------------------------------------------------------
# Auto-derived structure_type candidates
# ------------------------------------------------------------------

# (substring-in-lowercase-body) -> structure_type vocabulary item
_STRUCTURE_KEYWORD_MAP: list[tuple[str, str]] = [
    ("drain system", "drain_system"),
    ("drain ", "drain_system"),
    ("creek mouth", "creek_mouth_system"),
    ("creek-mouth", "creek_mouth_system"),
    ("creek mouth system", "creek_mouth_system"),
    ("creek junction", "creek_mouth_system"),
    ("oyster", "oyster_bar"),
    ("seagrass", "seagrass_patch"),
    ("sand island", "sand_lobe"),
    ("sand lobe", "sand_lobe"),
    ("sand/mud", "shallow_flat"),
    ("shallow flat", "shallow_flat"),
    ("shoreline bend", "shoreline_bend"),
    ("shoreline cut", "shoreline_cut"),
    ("current split", "current_split"),
    ("current-split", "current_split"),
    ("island chain", "island"),
    ("island ", "island"),
    ("islet", "island"),
    ("trough", "trough"),
    ("pass throat", "pass"),
    ("pass-throat", "pass"),
    ("pass ", "pass"),
    ("cove", "cove"),
    ("pocket", "cove"),
    ("peninsula", "mangrove_peninsula"),
    ("island edge", "island_edge"),
    ("island-edge", "island_edge"),
    ("point", "point"),
]


def derive_structure_candidates(body: str) -> list[str]:
    """Best-effort keyword sweep over the prose body. Order-preserving + dedup.

    This is informational metadata. The user's GT-usage instructions
    (`ground_truth/anchors/Purpose of ground truth.txt`) say to score by
    coverage (geometry), not by language match. structure_type candidates
    are kept on the GT entry so reports can flag flow-driver / feature-type
    misreads as required, not for primary anchor matching."""
    body_low = body.lower()
    seen: list[str] = []
    for kw, sty in _STRUCTURE_KEYWORD_MAP:
        if kw in body_low and sty not in seen:
            seen.append(sty)
    return seen


# ------------------------------------------------------------------
# Red-box detection (carried over from the original)
# ------------------------------------------------------------------


def detect_red_boxes(image_path: Path, min_area: int = 600) -> list[tuple[int, int, int, int]]:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red_mask = (r > 180) & (g < 100) & (b < 100)

    h, w = red_mask.shape
    seen = np.zeros_like(red_mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y0 in range(h):
        for x0 in range(w):
            if not red_mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            min_x, min_y, max_x, max_y = x0, y0, x0, y0
            count = 0
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if seen[y, x] or not red_mask[y, x]:
                    continue
                seen[y, x] = True
                count += 1
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                for dy, dx in nbrs:
                    stack.append((y + dy, x + dx))
            if count >= min_area:
                boxes.append((min_x, min_y, max_x, max_y))
    return boxes


# ------------------------------------------------------------------
# Geo
# ------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def pixel_centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ------------------------------------------------------------------
# Per-cell build
# ------------------------------------------------------------------


def build_one(spec: GTCellSpec, write: bool) -> int:
    print()
    print(f"=== {spec.cell_id} ===")
    if not spec.z16_source_image.exists():
        print(f"  ERROR: z16 source image missing: {spec.z16_source_image}")
        return 2

    prose = parse_prose(spec.prose_txt.read_text(encoding="utf-8"))
    if not prose:
        print(f"  ERROR: no anchors parsed from {spec.prose_txt}")
        return 2

    center_lat, center_lon = _sub_cell_center(spec.parent_id, spec.cell_num)

    with Image.open(spec.annotated_png) as im:
        if im.size != (IMG_SIZE_PX, IMG_SIZE_PX):
            print(f"  ERROR: annotated image dims {im.size} != {(IMG_SIZE_PX, IMG_SIZE_PX)}")
            return 2

    boxes = detect_red_boxes(spec.annotated_png)
    print(f"  parsed {len(prose)} prose anchors, detected {len(boxes)} red boxes")
    if len(boxes) != len(prose):
        print("  WARNING: detected box count != prose count; "
              "matching by nearest may produce odd assignments.")

    # Detected box info
    detected = []
    for bbox in boxes:
        cx, cy = pixel_centroid(bbox)
        clat, clon = pixel_to_latlon(cx, cy, IMG_SIZE_PX, center_lat, center_lon, ZOOM)
        detected.append({
            "bbox": bbox,
            "bbox_centroid_px": (cx, cy),
            "bbox_centroid_latlon": (clat, clon),
        })

    # Per-prose-anchor: pick containing box; fall back to nearest centroid.
    remaining = list(range(len(detected)))
    assignments: dict[int, int] = {}
    for p in prose:
        if not remaining:
            print(f"  WARN: prose anchor #{p.num} has no detected box left.")
            continue
        prose_px, prose_py = latlon_to_pixel(
            p.lat, p.lon, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        containing = [
            di for di in remaining
            if detected[di]["bbox"][0] <= prose_px <= detected[di]["bbox"][2]
            and detected[di]["bbox"][1] <= prose_py <= detected[di]["bbox"][3]
        ]
        if containing:
            best_idx = min(containing, key=lambda di: (
                (detected[di]["bbox"][2] - detected[di]["bbox"][0])
                * (detected[di]["bbox"][3] - detected[di]["bbox"][1])
            ))
        else:
            best_idx = min(
                remaining,
                key=lambda di: haversine_m(
                    p.lat, p.lon, *detected[di]["bbox_centroid_latlon"],
                ),
            )
            print(f"  NOTE: prose #{p.num} ({p.label_prose[:40]!r}) latlon "
                  f"falls outside any detected box; assigning nearest by centroid.")
        assignments[p.num] = best_idx
        remaining.remove(best_idx)

    overrides = _PER_CELL_OVERRIDES.get(spec.cell_id, {})

    # Build anchors[]
    anchors_out = []
    rt_failures = 0
    for p in prose:
        if p.num not in assignments:
            continue
        d = detected[assignments[p.num]]
        bbox = list(map(int, d["bbox"]))
        # pixel_center comes from prose latlon (round-trips perfectly).
        px, py = latlon_to_pixel(
            p.lat, p.lon, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        rt_lat, rt_lon = pixel_to_latlon(px, py, IMG_SIZE_PX, center_lat, center_lon, ZOOM)
        rt_err = haversine_m(p.lat, p.lon, rt_lat, rt_lon)
        if rt_err > 5.0:
            rt_failures += 1
            print(f"  WARN: gt{p.num} round-trip err {rt_err:.2f}m > 5m")

        ov = overrides.get(p.num, {})
        tier = ov.get("tier_override", p.tier)
        structure_types = ov.get("structure_type_candidates") or ov.get("structure_type_options") \
            or derive_structure_candidates(p.description)
        status = ov.get("status", "active")
        expects_dz = ov.get("expected_needs_deeper_zoom", False)

        entry = {
            "gt_id": f"gt{p.num}",
            "label": ov.get("label", p.label_prose),
            "label_prose": p.label_prose,
            "tier": tier,
            "tier_note": p.tier_note,
            "structure_type_candidates": structure_types,
            "status": status,
            "expected_needs_deeper_zoom": expects_dz,
            "pixel_center": [round(px, 1), round(py, 1)],
            "pixel_bbox": bbox,
            "latlon_center": [p.lat, p.lon],
            "description": p.description,
        }
        if "review_note" in ov:
            entry["review_note"] = ov["review_note"]
        anchors_out.append(entry)

    if rt_failures:
        print(f"  ERROR: {rt_failures} round-trip errors > 5m")
        return 2

    # Tier histogram for the report
    tier_counts: dict[str, int] = {}
    for a in anchors_out:
        key = f"T{a['tier']}" if a["tier"] is not None else "T?"
        tier_counts[key] = tier_counts.get(key, 0) + 1
    print(f"  anchors: {len(anchors_out)}  tiers: " +
          "  ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())))

    payload = {
        "cell_id": spec.cell_id,
        "z16_image": str(spec.z16_source_image.relative_to(REPO_ROOT)).replace("\\", "/"),
        "z16_center": [center_lat, center_lon],
        "image_size_px": [IMG_SIZE_PX, IMG_SIZE_PX],
        "zoom": ZOOM,
        "scale": 2,
        "source": {
            "annotated_image": str(spec.annotated_png.relative_to(REPO_ROOT)).replace("\\", "/"),
            "prose": str(spec.prose_txt.relative_to(REPO_ROOT)).replace("\\", "/"),
            "build_script": "scripts/build_gt_anchors.py",
        },
        "anchors": anchors_out,
    }

    if write:
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = spec.out_dir / "gt_anchors.json"
        out_overlay = spec.out_dir / "gt_overlay.png"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not out_overlay.exists() or out_overlay.stat().st_size != spec.annotated_png.stat().st_size:
            shutil.copy2(spec.annotated_png, out_overlay)
        print(f"  wrote {out_json.relative_to(REPO_ROOT)}")
    else:
        print(f"  --check-only: not writing.")

    return 0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", default=None,
                        help="Restrict to a specific cell_id (repeatable). "
                             "Default: every cell with a GT pair.")
    parser.add_argument("--check-only", action="store_true",
                        help="Don't write gt_anchors.json; just report.")
    args = parser.parse_args()

    cells = discover_gt_cells()
    if args.cell:
        wanted = set(args.cell)
        cells = [c for c in cells if c.cell_id in wanted]
        missing = wanted - {c.cell_id for c in cells}
        if missing:
            print(f"WARN: requested cells not found: {sorted(missing)}", file=sys.stderr)
    if not cells:
        print("no cells to process.", file=sys.stderr)
        return 2

    print(f"processing {len(cells)} cell(s) (write={not args.check_only}):")
    for c in cells:
        print(f"  {c.cell_id}  ({c.annotated_png.name}, {c.prose_txt.name})")

    overall = 0
    for c in cells:
        rc = build_one(c, write=not args.check_only)
        if rc != 0:
            overall = rc

    print()
    print(f"done. exit={overall}")
    return overall


if __name__ == "__main__":
    sys.exit(main())
