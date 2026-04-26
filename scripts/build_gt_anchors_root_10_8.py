"""Build the pixel-precise ground-truth anchor file for root-10-8.

One-shot data-prep script for Phase C TASK-0. Reads the hand-labeled
annotated PNG (red bounding boxes on the z16 image) and the prose anchor
descriptions, detects each box by color, matches it to a prose entry by
nearest haversine distance, and writes the canonical
`gt_anchors.json` consumed by downstream Phase C tooling (TASK-2 coord-gen
harness in particular).

Per the addendum in `docs/PHASE_C_TASKS.md`, each GT entry carries a
`status` field. Match-scoring excludes anything that isn't `"active"`.

Usage:
  python scripts/build_gt_anchors_root_10_8.py
  python scripts/build_gt_anchors_root_10_8.py --check-only   # don't write

Inputs:
  - ground_truth/anchors/z0_10_8_anchors.png   (your labeled image)
  - ground_truth/anchors/z0_10_8_anchors.txt   (your prose with lat/lons)

Outputs:
  - data/areas/rookery_bay_v2/images/structures/root-10-8/gt_anchors.json
  - data/areas/rookery_bay_v2/images/structures/root-10-8/gt_overlay.png
    (copy of the annotated PNG — keeps everything for that cell in one place)
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

# ------------------------------------------------------------------
# Bootstrap so we can import readwater + scripts/_cells.
# ------------------------------------------------------------------

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
WORKTREE_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(WORKTREE_ROOT / "src"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts"))

from _cells import _sub_cell_center  # noqa: E402
from readwater.pipeline.structure.geo import (  # noqa: E402
    latlon_to_pixel,
    pixel_to_latlon,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

GT_DIR = REPO_ROOT / "ground_truth" / "anchors"
GT_IMAGE = GT_DIR / "z0_10_8_anchors.png"
GT_TXT = GT_DIR / "z0_10_8_anchors.txt"

OUT_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images" / "structures" / "root-10-8"
OUT_JSON = OUT_DIR / "gt_anchors.json"
OUT_OVERLAY = OUT_DIR / "gt_overlay.png"

CELL_ID = "root-10-8"
ZOOM = 16
IMG_SIZE_PX = 1280

# ------------------------------------------------------------------
# Per-anchor policy (from docs/PHASE_C_TASKS.md addendum)
# ------------------------------------------------------------------

# Maps the `Anchor N` / `Box N` numbers used in the prose to a richer
# spec than the prose alone provides. structure_type_options drives match
# scoring (any of these counts as a hit).
GT_SPEC: dict[int, dict] = {
    1: {
        "label": "NW drain system",
        "structure_type_options": ["drain_system", "creek_mouth_system"],
        "tier": 1,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
    2: {
        "label": "Small mangrove island (W lagoon, upper)",
        "structure_type_options": ["island"],
        "tier": 1,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
    3: {
        "label": "Elongated mangrove islet (W lagoon, lower)",
        "structure_type_options": ["island"],
        "tier": 1,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
    4: {
        "label": "SE hammock island",
        "structure_type_options": ["island"],
        "tier": 1,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
    5: {
        "label": "Peninsula point (N tip of southern peninsula)",
        "structure_type_options": ["point"],
        "tier": 3,
        "status": "under_review",
        "expected_needs_deeper_zoom": False,
        "review_note": (
            "Classification debatable; user research pending. Excluded "
            "from match scoring until resolved."
        ),
    },
    6: {
        "label": "E-shore trough",
        "structure_type_options": ["trough", "current_split"],
        "tier": 3,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
    7: {
        "label": "Seagrass / sand / flat lobe (ambiguous)",
        "structure_type_options": ["seagrass_patch", "sand_lobe", "shallow_flat"],
        "tier": None,
        "status": "active",
        "expected_needs_deeper_zoom": True,
        "review_note": (
            "Ambiguous bottom type at z16. v3 should mark "
            "needs_deeper_zoom: true. Any of the three structure types is a hit."
        ),
    },
    8: {
        "label": "Central junction island",
        "structure_type_options": ["island"],
        "tier": 1,
        "status": "active",
        "expected_needs_deeper_zoom": False,
    },
}


# ------------------------------------------------------------------
# Prose parsing
# ------------------------------------------------------------------

@dataclass
class ProseAnchor:
    num: int           # 1..8 from the txt file's "Anchor N" / "Box N"
    label_prose: str   # the descriptor after the dash on the header line
    lat: float
    lon: float
    description: str   # the body paragraph below the header line


_HEADER_RE = re.compile(
    r"^\s*(?:Anchor|Box)\s+(\d+)\s*[—\-]\s*(.+?)\s+(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$"
)


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
        out.append(ProseAnchor(num=num, label_prose=label, lat=lat, lon=lon, description=body))
        i += 2
    return out


# ------------------------------------------------------------------
# Red-box detection
# ------------------------------------------------------------------


def detect_red_boxes(image_path: Path, min_area: int = 600) -> list[tuple[int, int, int, int]]:
    """Return a list of (x0, y0, x1, y1) for each red rectangle outline.

    The annotation tool draws hollow red rectangles. We:
      1. Threshold pixels where R is high and G,B are low (pure red).
      2. Find connected components.
      3. For each big-enough component, take its axis-aligned bbox.

    Box 6 in this GT is rotated (tilted along the shoreline) — its
    axis-aligned bbox over-covers slightly. That's acceptable for GT;
    coord-gen is asked to predict tight axis-aligned bboxes and we'll
    score IoU between the two.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    red_mask = (r > 180) & (g < 100) & (b < 100)

    # Tiny in-house connected-components (no scipy dep): flood-fill via BFS.
    h, w = red_mask.shape
    seen = np.zeros_like(red_mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    # 8-connectivity neighborhood
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y0 in range(h):
        for x0 in range(w):
            if not red_mask[y0, x0] or seen[y0, x0]:
                continue
            # BFS this component
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
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print the verification table but don't write gt_anchors.json",
    )
    args = parser.parse_args()

    # 1. Inputs
    if not GT_IMAGE.exists():
        print(f"missing GT image: {GT_IMAGE}", file=sys.stderr)
        return 2
    if not GT_TXT.exists():
        print(f"missing GT prose: {GT_TXT}", file=sys.stderr)
        return 2

    prose = parse_prose(GT_TXT.read_text(encoding="utf-8"))
    if {p.num for p in prose} != set(GT_SPEC.keys()):
        print(
            f"prose anchor numbers {sorted(p.num for p in prose)} don't match "
            f"GT_SPEC keys {sorted(GT_SPEC.keys())}",
            file=sys.stderr,
        )
        return 2
    prose_by_num = {p.num: p for p in prose}

    # 2. Cell center via the canonical _cells helper
    center_lat, center_lon = _sub_cell_center("root-10", 8)
    print(f"root-10-8 center: ({center_lat:.6f}, {center_lon:.6f})")

    # Confirm image dims
    with Image.open(GT_IMAGE) as im:
        if im.size != (IMG_SIZE_PX, IMG_SIZE_PX):
            print(f"image dims {im.size} != expected {(IMG_SIZE_PX, IMG_SIZE_PX)}", file=sys.stderr)
            return 2

    # 3. Detect red boxes
    boxes = detect_red_boxes(GT_IMAGE)
    print(f"detected {len(boxes)} red boxes")
    if len(boxes) != len(GT_SPEC):
        print(
            f"WARNING: detected {len(boxes)} boxes but expected {len(GT_SPEC)}. "
            "Inspect annotated image — overlapping boxes or non-red text may "
            "be confusing the detector."
        )

    # 4. Match each prose anchor to its nearest detected box.
    #
    # Each prose anchor's lat/lon is the user-declared structure center.
    # We project that to pixel space and pick the bbox that contains it (or
    # is closest if none contain it — flagged for manual review).
    #
    # Pixel_center is then derived from the prose lat/lon via latlon_to_pixel
    # (NOT from the bbox centroid), so pixel_center and latlon_center are
    # guaranteed to round-trip through the geo helpers.
    detected: list[dict] = []
    for bbox in boxes:
        cx, cy = pixel_centroid(bbox)
        clat, clon = pixel_to_latlon(cx, cy, IMG_SIZE_PX, center_lat, center_lon, ZOOM)
        detected.append({
            "bbox": bbox,
            "bbox_centroid_px": (cx, cy),
            "bbox_centroid_latlon": (clat, clon),
        })

    # For each prose anchor: project its latlon to a pixel; find the box that
    # contains that pixel (preferred), else the box whose centroid is nearest.
    remaining = list(range(len(detected)))
    assignments: dict[int, int] = {}      # prose_num -> detected_idx
    centroid_offsets: dict[int, float] = {}  # diagnostic only
    for num in sorted(GT_SPEC.keys()):
        p = prose_by_num[num]
        prose_px, prose_py = latlon_to_pixel(
            p.lat, p.lon, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        # First pass: find boxes containing the prose pixel.
        containing = [
            di for di in remaining
            if detected[di]["bbox"][0] <= prose_px <= detected[di]["bbox"][2]
            and detected[di]["bbox"][1] <= prose_py <= detected[di]["bbox"][3]
        ]
        if containing:
            # If multiple boxes contain the pixel, pick the smallest (most
            # specific). With this GT none should overlap, but safe.
            def _area(di: int) -> int:
                b = detected[di]["bbox"]
                return (b[2] - b[0]) * (b[3] - b[1])
            best_idx = min(containing, key=_area)
        else:
            # Fallback: nearest box centroid by haversine.
            best_idx = min(
                remaining,
                key=lambda di: haversine_m(
                    p.lat, p.lon, *detected[di]["bbox_centroid_latlon"],
                ),
            )
            print(
                f"WARNING: prose anchor {num} latlon doesn't fall inside any "
                f"detected box; using nearest by centroid distance."
            )
        assignments[num] = best_idx
        # Diagnostic: how far is prose latlon from this box's geometric center?
        centroid_offsets[num] = haversine_m(
            p.lat, p.lon, *detected[best_idx]["bbox_centroid_latlon"],
        )
        remaining.remove(best_idx)

    # 5. Verification table
    print()
    print(
        f"{'#':>2}  {'label':<48s}  {'bbox':<24s}  "
        f"{'prose-vs-bbox-ctr (m)':>22s}"
    )
    print("-" * 104)
    for num in sorted(assignments.keys()):
        d = detected[assignments[num]]
        bbox = d["bbox"]
        off = centroid_offsets[num]
        # Informational only — irregular shapes (rotated trough, multi-clump
        # islands) can put the user's true center far from the bbox centroid.
        print(
            f"{num:>2}  {GT_SPEC[num]['label'][:48]:<48s}  "
            f"({bbox[0]:>4d},{bbox[1]:>4d},{bbox[2]:>4d},{bbox[3]:>4d})  "
            f"{off:>22.1f}"
        )

    # 6. Acceptance: round-trip pixel_center <-> latlon_center within 5 m.
    # pixel_center comes from latlon_to_pixel(prose); converting back must
    # match the prose latlon.
    print()
    print("round-trip check (pixel_center -> latlon_center within 5 m):")
    rt_failures = 0
    for num in sorted(assignments.keys()):
        p = prose_by_num[num]
        px, py = latlon_to_pixel(
            p.lat, p.lon, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        rt_lat, rt_lon = pixel_to_latlon(
            px, py, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        err_m = haversine_m(p.lat, p.lon, rt_lat, rt_lon)
        ok = err_m < 5.0
        if not ok:
            rt_failures += 1
        print(
            f"  #{num} latlon ({p.lat:.6f},{p.lon:.6f}) -> px ({px:.1f},{py:.1f}) "
            f"-> latlon ({rt_lat:.6f},{rt_lon:.6f})  err={err_m:.4f}m  "
            f"[{'OK' if ok else 'FAIL'}]"
        )
    if rt_failures:
        print(f"FAIL: {rt_failures} round-trip errors >5 m.")
        return 2

    # 7. Build the gt_anchors.json structure.
    z16_image_relpath = "data/areas/rookery_bay_v2/images/z0_10_8.png"
    anchors_out = []
    for num in sorted(GT_SPEC.keys()):
        p = prose_by_num[num]
        d = detected[assignments[num]]
        bbox = list(map(int, d["bbox"]))
        # pixel_center comes from the prose latlon (the human-declared
        # structure center), not the bbox centroid. This is what coord-gen
        # is asked to predict.
        px, py = latlon_to_pixel(
            p.lat, p.lon, IMG_SIZE_PX, center_lat, center_lon, ZOOM,
        )
        spec = GT_SPEC[num]
        entry = {
            "gt_id": f"gt{num}",
            "label": spec["label"],
            "label_prose": p.label_prose,
            "structure_type_options": spec["structure_type_options"],
            "tier": spec["tier"],
            "status": spec["status"],
            "expected_needs_deeper_zoom": spec["expected_needs_deeper_zoom"],
            "pixel_center": [round(px, 1), round(py, 1)],
            "pixel_bbox": bbox,
            "latlon_center": [p.lat, p.lon],
            "description": p.description,
        }
        if "review_note" in spec:
            entry["review_note"] = spec["review_note"]
        anchors_out.append(entry)

    payload = {
        "cell_id": CELL_ID,
        "z16_image": z16_image_relpath,
        "z16_center": [center_lat, center_lon],
        "image_size_px": [IMG_SIZE_PX, IMG_SIZE_PX],
        "zoom": ZOOM,
        "scale": 2,
        "source": {
            "annotated_image": "ground_truth/anchors/z0_10_8_anchors.png",
            "prose": "ground_truth/anchors/z0_10_8_anchors.txt",
            "build_script": "scripts/build_gt_anchors_root_10_8.py",
        },
        "anchors": anchors_out,
    }

    if args.check_only:
        print()
        print("--check-only: not writing.")
        print(json.dumps(payload, indent=2)[:500] + "\n  ...")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not OUT_OVERLAY.exists() or OUT_OVERLAY.stat().st_size != GT_IMAGE.stat().st_size:
        shutil.copy2(GT_IMAGE, OUT_OVERLAY)

    print()
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_OVERLAY.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
