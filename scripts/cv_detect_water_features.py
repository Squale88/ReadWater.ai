"""CV-based water-feature CANDIDATE detection at the medium scale.

Phase-1: find narrow water strips ("constrictions") in the water mask at
scale R=25 (catches throats up to ~50 px wide), then keep only those that
match one of four rules:

  DRAIN         — both adjacent wide-water sides are substantial
                  (side_b >= DRAIN_MIN_SIDE_B_AREA)
  CREEK_MOUTH   — one big bay, throat >= 40 px, water continues inland
                  (low inland-water compactness — long thin shape)
  LARGE_POCKET  — one big bay, throat >= 40 px, dead-end indent
                  (high inland-water compactness — roundish shape)
  SHOAL         — one big bay, throat 20-40 px, surrounded by water
                  (high water density in candidate neighborhood)

Anything else (small mangrove pockets, interior creek sections, edge artifacts,
mask noise) is dropped. The downstream LLM verifier still owns the final
call on each kept candidate; the rules above just keep the input set focused
on features that are likely to be anchors.

Algorithm:
  1. Smooth the water mask (boundary cleanup, tiny kernel).
  2. wide = open(water, R). narrow = water - wide.
  3. Connected components on `narrow` (above noise floor) and `wide`.
  4. For each narrow CC: measure throat width, two largest adjacent wide
     areas, water density in neighborhood, and (for >=40px throats) inland
     compactness.
  5. Apply classify(); drop everything that doesn't match.

Render kept candidates color-coded by category on the satellite.

Input:  data/areas/rookery_bay_v2_google_water/<cell>_water_mask.png
Output: data/areas/rookery_bay_v2/images/structures/<cell>/cv_phase1_candidates_<ts>.{png,json}

Usage:
  python scripts/cv_detect_water_features.py --cell root-10-8
  python scripts/cv_detect_water_features.py --cell root-2-9 --cell root-11-1 --cell root-11-5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- Phase 1 (medium scale) parameters ----

SCALE_RADIUS = 25          # captures throats up to ~50 px wide (medium scale)
MIN_NARROW_AREA = 60       # drop narrow CCs smaller than this (mask-noise floor)
MIN_WIDE_AREA = 200        # drop wide CCs smaller than this when looking for "sides"
SMOOTH_RADIUS = 4          # boundary smoothing kernel for the water mask
EDGE_MARGIN_PX = 6         # bbox within this many pixels of the frame -> is_edge_truncated

# ---- 4-category classification thresholds ----
# (derived from manual classification of root-11-5; revisit on other cells)

DRAIN_MIN_SIDE_B_AREA = 500       # second-side area required to call a feature a DRAIN
MOUTH_MIN_SIDE_A_AREA = 5_000     # first-side area required to call ANY single-side feature
MOUTH_MIN_THROAT_PX = 40          # throat width to qualify as a CREEK MOUTH or LARGE_POCKET
SHOAL_MIN_THROAT_PX = 20          # lower bound for SHOAL throat (anything below is too small)
SHOAL_MAX_THROAT_PX = 40          # upper bound (above goes into MOUTH/POCKET category)
SHOAL_NEIGHBORHOOD_RADIUS = 100   # px radius around the candidate to measure water density
SHOAL_MIN_WATER_DENSITY = 0.65    # neighborhood >= this fraction water -> SHOAL

# CREEK_MOUTH vs LARGE_POCKET split: both have one substantial wide side and a
# throat >= 40 px. The difference is the SHAPE of the inland water reachable
# from the throat:
#   - creek mouth: water snakes inland for some distance (low compactness =
#     long thin shape, fills little of its bbox)
#   - large pocket: water fills a roundish bay (high compactness = roundish
#     shape, fills most of its bbox)
# Threshold derived from one cell (root-11-5 c15 = 0.302 vs pockets 0.37-0.59).
CREEK_MOUTH_MAX_INLAND_COMPACTNESS = 0.35


# ---- Pure-numpy morphology ----


def erode_4conn(mask: np.ndarray, iterations: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(iterations):
        padded = np.pad(m, 1, constant_values=False)
        m = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
    return m


def dilate_4conn(mask: np.ndarray, iterations: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(iterations):
        padded = np.pad(m, 1, constant_values=False)
        m = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
        )
    return m


def open_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening = erode then dilate. Removes features < 2*radius wide."""
    return dilate_4conn(erode_4conn(mask, radius), radius)


def smooth_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Closing then opening at small radius to clean noisy mask boundaries."""
    closed = erode_4conn(dilate_4conn(mask, radius), radius)
    return dilate_4conn(erode_4conn(closed, radius), radius)


def connected_components(mask: np.ndarray, min_pixels: int = 1) -> list[dict]:
    """4-connected CC scan. Each comp: {bbox, center, area, pixels (set of (y,x))}."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            min_x, min_y, max_x, max_y = x0, y0, x0, y0
            sum_x = sum_y = 0
            count = 0
            pixels: set[tuple[int, int]] = set()
            while stack:
                yy, xx = stack.pop()
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if visited[yy, xx] or not mask[yy, xx]:
                    continue
                visited[yy, xx] = True
                count += 1
                sum_x += xx
                sum_y += yy
                pixels.add((yy, xx))
                if xx < min_x: min_x = xx
                if xx > max_x: max_x = xx
                if yy < min_y: min_y = yy
                if yy > max_y: max_y = yy
                for dy, dx in nbrs:
                    stack.append((yy + dy, xx + dx))
            if count >= min_pixels:
                out.append({
                    "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                    "center": (sum_x / count, sum_y / count),
                    "area": count,
                    "pixels": pixels,
                })
    return out


def find_adjacent(narrow_pixels: set[tuple[int, int]],
                  wide_components: list[dict],
                  shape: tuple[int, int]) -> list[dict]:
    """Return wide CCs that touch (within 1 px of) the narrow region."""
    h, w = shape
    cand_mask = np.zeros((h, w), dtype=bool)
    for (y, x) in narrow_pixels:
        cand_mask[y, x] = True
    dilated = dilate_4conn(cand_mask, 1)
    adjacent = []
    for comp in wide_components:
        x0, y0, x1, y1 = comp["bbox"]
        comp_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for (y, x) in comp["pixels"]:
            comp_mask[y - y0, x - x0] = True
        if (comp_mask & dilated[y0:y1, x0:x1]).any():
            adjacent.append(comp)
    return adjacent


# ---- Detection ----


def water_density_around(water: np.ndarray,
                         center: tuple[float, float],
                         radius: int) -> float:
    """Fraction of pixels in a square neighborhood that are water.

    Used to distinguish SHOAL (constriction in open water — dense water around)
    from a tight constriction in mostly-mangrove context.
    """
    h, w = water.shape
    cx, cy = int(center[0]), int(center[1])
    x0 = max(0, cx - radius); x1 = min(w, cx + radius)
    y0 = max(0, cy - radius); y1 = min(h, cy + radius)
    region = water[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def inland_compactness(smoothed_water: np.ndarray,
                       narrow_pixels: set[tuple[int, int]],
                       side_a_pixels: set[tuple[int, int]]) -> float:
    """Compactness (area / bbox_area) of the water region that's inland of
    the throat — i.e., reachable from the narrow CC but excluding side_a's
    main wide CC.

    A CREEK MOUTH has water that snakes inland through mangrove (long thin
    shape -> low compactness). A LARGE POCKET has water that fills a roundish
    bay (compact shape -> high compactness).

    Returns 0.0 when there's no measurable inland water (the throat is purely
    on side_a's edge with nothing behind it — these get the lowest compactness
    and would be miscategorized as creek mouths; in practice they're rare
    because side_a + the narrow strip already filled the area).
    """
    h, w = smoothed_water.shape
    # Build a "candidate inland" mask = water minus side_a's pixels
    side_a_mask = np.zeros((h, w), dtype=bool)
    for (y, x) in side_a_pixels:
        side_a_mask[y, x] = True
    inland = smoothed_water & ~side_a_mask

    # BFS from any pixel of the narrow CC; if it's not in inland (could happen
    # if narrow is entirely contained in side_a, edge case), return 0.0
    seed_y, seed_x = next(iter(narrow_pixels))
    if not inland[seed_y, seed_x]:
        return 0.0

    visited = np.zeros_like(inland, dtype=bool)
    stack = [(seed_y, seed_x)]
    min_x = max_x = seed_x
    min_y = max_y = seed_y
    count = 0
    while stack:
        yy, xx = stack.pop()
        if yy < 0 or yy >= h or xx < 0 or xx >= w:
            continue
        if visited[yy, xx] or not inland[yy, xx]:
            continue
        visited[yy, xx] = True
        count += 1
        if xx < min_x: min_x = xx
        if xx > max_x: max_x = xx
        if yy < min_y: min_y = yy
        if yy > max_y: max_y = yy
        stack.extend([(yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)])

    bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    return count / max(1, bbox_area)


def classify(c: dict) -> str | None:
    """Apply the 4-category filter. Returns one of:
      DRAIN, CREEK_MOUTH, LARGE_POCKET, SHOAL
    or None if the candidate doesn't fit any kept category.
    """
    side_a = c["side_a_area_px"]
    side_b = c["side_b_area_px"]
    throat = c["throat_width_px"]
    density = c["water_density"]
    compactness = c.get("inland_compactness", 0.0)

    if side_b >= DRAIN_MIN_SIDE_B_AREA:
        return "DRAIN"
    if side_a < MOUTH_MIN_SIDE_A_AREA:
        return None
    # one-side-substantial cases
    if throat >= MOUTH_MIN_THROAT_PX:
        # creek mouth = water continues inland (long thin inland region)
        # large pocket = water fills a dead-end bay (compact inland region)
        if compactness < CREEK_MOUTH_MAX_INLAND_COMPACTNESS:
            return "CREEK_MOUTH"
        return "LARGE_POCKET"
    if SHOAL_MIN_THROAT_PX <= throat < MOUTH_MIN_THROAT_PX:
        if density >= SHOAL_MIN_WATER_DENSITY:
            return "SHOAL"
        # otherwise it's a small pocket / mangrove edge bump — drop
    return None


def detect_constrictions(water: np.ndarray,
                         scale_radius: int = SCALE_RADIUS,
                         min_throat_px: int = 0) -> tuple[list[dict], np.ndarray]:
    """One-pass constriction detection at a given scale.

    Returns (kept_candidates, wide_mask).

    Args:
      water: HxW boolean array (True = water).
      scale_radius: morphological opening radius. Catches throats up to ~2R.
      min_throat_px: drop narrow CCs whose short bbox dim is below this. Used
        when running multi-scale to avoid re-detecting features already caught
        by a smaller-scale pass.

    Detection pipeline:
      1. wide = open(water, R); narrow = water - wide
      2. For each narrow CC: measure throat width, side areas, density,
         and (for >=40px throats) inland compactness.
      3. Apply classify() to assign one of {DRAIN, CREEK_MOUTH, LARGE_POCKET, SHOAL}.
      4. Drop candidates that don't match any kept category.
    """
    smoothed = smooth_mask(water, radius=SMOOTH_RADIUS)
    wide = open_mask(smoothed, scale_radius)
    narrow = smoothed & ~wide

    narrow_ccs = connected_components(narrow, min_pixels=MIN_NARROW_AREA)
    wide_ccs = connected_components(wide, min_pixels=MIN_WIDE_AREA)

    kept: list[dict] = []
    for nc in narrow_ccs:
        bbox = nc["bbox"]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        throat_width = min(bw, bh)

        if throat_width < min_throat_px:
            continue   # already caught by a smaller-scale pass

        adj = find_adjacent(nc["pixels"], wide_ccs, water.shape)
        adj.sort(key=lambda c: c["area"], reverse=True)
        side_a_cc = adj[0] if len(adj) >= 1 else None
        side_a = side_a_cc["area"] if side_a_cc else 0
        side_b = adj[1]["area"] if len(adj) >= 2 else 0

        density = water_density_around(smoothed, nc["center"], SHOAL_NEIGHBORHOOD_RADIUS)

        # Compactness only matters for the >=40px creek/pocket split. Skip
        # the BFS for candidates that won't reach the creek/pocket branch.
        if side_a_cc is not None and throat_width >= MOUTH_MIN_THROAT_PX and side_b < DRAIN_MIN_SIDE_B_AREA:
            compactness = inland_compactness(smoothed, nc["pixels"], side_a_cc["pixels"])
        else:
            compactness = 0.0

        # Flag candidates whose bbox sits within EDGE_MARGIN_PX of any frame
        # edge. Their classification is suspect because the side-areas, water
        # density, and inland compactness are all computed from the on-frame
        # half of the feature only — the off-frame half is invisible. The
        # downstream LLM verifier (which has the wide z14/z13 context) is the
        # right place to make the final call on these.
        is_edge = (
            bbox[0] <= EDGE_MARGIN_PX
            or bbox[1] <= EDGE_MARGIN_PX
            or bbox[2] >= water.shape[1] - EDGE_MARGIN_PX
            or bbox[3] >= water.shape[0] - EDGE_MARGIN_PX
        )

        cand = {
            "bbox": bbox,
            "center": nc["center"],
            "narrow_area_px": nc["area"],
            "throat_width_px": throat_width,
            "side_a_area_px": side_a,
            "side_b_area_px": side_b,
            "n_adjacent_wides": len(adj),
            "water_density": round(density, 3),
            "inland_compactness": round(compactness, 3),
            "scale_radius_px": scale_radius,
            "is_edge_truncated": is_edge,
        }
        cat = classify(cand)
        if cat is None:
            continue
        cand["category"] = cat
        kept.append(cand)

    return kept, wide


# ---- Edge candidate recomputation using wide z14 mask ----


def _z16_to_z14(z16_x: float, z16_y: float) -> tuple[float, float]:
    """Map z16 pixel coords to z14 pixel coords.

    Both Static-Maps tiles are 1280x1280 centered on the same lat/lon.
    z14 covers 4x the linear extent (16x area), so the z16 cell footprint
    occupies the central 320x320 region of the z14 image (pixels 480..800
    in each axis).
    """
    return (480.0 + z16_x / 4.0, 480.0 + z16_y / 4.0)


def _water_mask_from_styled_png(styled_png: Path) -> np.ndarray:
    """Threshold a styled water tile (water=blue, land=white) into a binary
    boolean mask. Mirrors scripts/google_water_mask.py:water_mask_from_styled.
    """
    img = Image.open(styled_png).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (b > 128) & (r < 96) & (g < 96)


def _wide_ccs_overlapping(z14_bbox: tuple[int, int, int, int],
                          wide_ccs: list[dict],
                          shape: tuple[int, int]) -> list[dict]:
    """Return wide-z14 CCs that overlap or touch (within 2 px of) the given
    z14 bounding box. Used to find side_a / side_b at z14 for an edge
    candidate whose z16 throat region maps to this z14 box.
    """
    h, w = shape
    x0, y0, x1, y1 = z14_bbox
    x0 = max(0, x0 - 2); y0 = max(0, y0 - 2)
    x1 = min(w, x1 + 2); y1 = min(h, y1 + 2)
    region = np.zeros((h, w), dtype=bool)
    region[y0:y1, x0:x1] = True

    adjacent = []
    for comp in wide_ccs:
        cx0, cy0, cx1, cy1 = comp["bbox"]
        if cx1 < x0 or cx0 > x1 or cy1 < y0 or cy0 > y1:
            continue   # bboxes don't overlap
        comp_mask = np.zeros((cy1 - cy0, cx1 - cx0), dtype=bool)
        for (y, x) in comp["pixels"]:
            comp_mask[y - cy0, x - cx0] = True
        if (comp_mask & region[cy0:cy1, cx0:cx1]).any():
            adjacent.append(comp)
    return adjacent


EDGE_REVALIDATE_MATCH_RADIUS_Z14 = 25   # z14 px (= ~100 z16 ground px) — matching tolerance
EDGE_REVALIDATE_MIN_NARROW_AREA_Z14 = 6 # min narrow CC pixels at z14 (1/16 area ratio of z16's 100)


def revalidate_edges_at_z14(candidates: list[dict], cell_id: str) -> tuple[int, int, int]:
    """Re-run the constriction discovery at z14 over the wider footprint, then
    DROP every is_edge_truncated z16 candidate that has no matching z14
    constriction near it.

    Insight: most z16 edge "features" aren't really features — they're
    artifacts of the cell boundary cutting through what is really open water,
    a continuous shoreline, or a mangrove edge. When you re-detect at z14
    with the wider context visible, those artifacts disappear. The ones that
    SURVIVE the wider-scale detection are real features whose throat happens
    to fall near the z16 cell border.

    Augments surviving candidates with `confirmed_at_z14: true` so the
    downstream consumer can see which edge candidates passed the wider check.

    Returns (n_edge_input, n_confirmed, n_dropped). No-ops (returns 0,0,0) if
    the wide z14 styled image isn't on disk for this cell.
    """
    edge_indices = [i for i, c in enumerate(candidates) if c.get("is_edge_truncated")]
    if not edge_indices:
        return (0, 0, 0)

    z14_styled = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2_google_water"
                  / f"{cell_id}_wide_z14_styled.png")
    if not z14_styled.exists():
        print(f"  no z14 wide tile for {cell_id} — skipping edge revalidation")
        return (0, 0, 0)

    z14_water = _water_mask_from_styled_png(z14_styled)
    if z14_water.shape != (1280, 1280):
        z14_img = Image.fromarray((z14_water.astype(np.uint8) * 255), mode="L")
        z14_img = z14_img.resize((1280, 1280), Image.NEAREST)
        z14_water = np.array(z14_img) > 127

    # Run the SAME wide/narrow split at z14, with R scaled to match the z16
    # ground scale (1 z14 px = 4 z16 ground px, so z14 R = z16 R / 4).
    smoothed_z14 = smooth_mask(z14_water, radius=SMOOTH_RADIUS)
    z14_R = max(1, SCALE_RADIUS // 4)            # = 6
    wide_z14 = open_mask(smoothed_z14, z14_R)
    narrow_z14 = smoothed_z14 & ~wide_z14
    z14_narrow_ccs = connected_components(
        narrow_z14, min_pixels=EDGE_REVALIDATE_MIN_NARROW_AREA_Z14,
    )

    drop_indices: list[int] = []
    n_confirmed = 0

    for i in edge_indices:
        c = candidates[i]
        z14_cx, z14_cy = _z16_to_z14(c["center"][0], c["center"][1])
        # Find the closest narrow z14 CC center; pass if within match radius
        match_dist_sq = EDGE_REVALIDATE_MATCH_RADIUS_Z14 ** 2
        matched = False
        for nc in z14_narrow_ccs:
            ncx, ncy = nc["center"]
            d_sq = (ncx - z14_cx) ** 2 + (ncy - z14_cy) ** 2
            if d_sq < match_dist_sq:
                matched = True
                break
        if matched:
            c["confirmed_at_z14"] = True
            n_confirmed += 1
        else:
            drop_indices.append(i)

    for i in sorted(drop_indices, reverse=True):
        candidates.pop(i)

    return (len(edge_indices), n_confirmed, len(drop_indices))


# ---- Rendering ----


def _grid_cell_for(px: float, py: float,
                   image_size: tuple[int, int] = (1280, 1280),
                   rows: int = 8, cols: int = 8) -> str:
    w, h = image_size
    col = max(0, min(cols - 1, int(px / (w / cols))))
    row = max(0, min(rows - 1, int(py / (h / rows))))
    return f"{chr(ord('A') + row)}{col + 1}"


def _load_font(size: int):
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


CATEGORY_COLOR = {
    "DRAIN":        (220,  30,  30, 255),    # red
    "CREEK_MOUTH":  (255, 165,   0, 255),    # orange — rare, true creek extending inland
    "LARGE_POCKET": (255, 230,  50, 255),    # yellow — common, dead-end indent
    "SHOAL":        ( 60, 200, 230, 255),    # cyan-ish — open-water constriction
}
WIDE_TINT = (80, 170, 220, 60)               # soft cyan tint to show wide-water bodies


def render_overlay(base_image_path: Path,
                   candidates: list[dict],
                   wide_mask: np.ndarray,
                   output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Tint the wide water bodies so reviewers can SEE which sides the
    # algorithm considered wide-water (helps interpret side_a / side_b areas).
    h, w = wide_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[wide_mask] = WIDE_TINT
    wide_layer = Image.fromarray(rgba, mode="RGBA")
    layer = Image.alpha_composite(layer, wide_layer)
    draw = ImageDraw.Draw(layer, "RGBA")

    # 8x8 A1-H8 grid
    rows, cols = 8, 8
    gw, gh = image_size
    cell_w, cell_h = gw / cols, gh / rows
    for i in range(1, cols):
        x = int(i * cell_w)
        draw.line([(x + 1, 0), (x + 1, gh)], fill=(0, 0, 0, 200), width=1)
        draw.line([(x, 0), (x, gh)], fill=(255, 255, 255, 230), width=2)
    for j in range(1, rows):
        y = int(j * cell_h)
        draw.line([(0, y + 1), (gw, y + 1)], fill=(0, 0, 0, 200), width=1)
        draw.line([(0, y), (gw, y)], fill=(255, 255, 255, 230), width=2)
    grid_font = _load_font(max(10, int(min(cell_w, cell_h) * 0.30)))
    for r in range(rows):
        for c in range(cols):
            text = f"{chr(ord('A') + r)}{c + 1}"
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            bbox_t = grid_font.getbbox(text)
            tw = bbox_t[2] - bbox_t[0]
            th = bbox_t[3] - bbox_t[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), text, fill="black", font=grid_font)
            draw.text((tx, ty), text, fill="white", font=grid_font)

    label_font = _load_font(15)
    for i, cand in enumerate(candidates, start=1):
        bbox = cand["bbox"]
        cx, cy = cand["center"]
        cell = _grid_cell_for(cx, cy, image_size)
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        x0 = max(0, bbox[0] - 4)
        y0 = max(0, bbox[1] - 4)
        x1 = min(image_size[0], bbox[2] + 4)
        y1 = min(image_size[1], bbox[3] + 4)
        is_edge = cand.get("is_edge_truncated", False)
        if is_edge:
            # Dashed outline for edge-truncated candidates so they're visually
            # distinct without changing their category color.
            for seg_x in range(x0, x1, 12):
                draw.line([(seg_x, y0), (min(seg_x + 6, x1), y0)], fill=color, width=3)
                draw.line([(seg_x, y1), (min(seg_x + 6, x1), y1)], fill=color, width=3)
            for seg_y in range(y0, y1, 12):
                draw.line([(x0, seg_y), (x0, min(seg_y + 6, y1))], fill=color, width=3)
                draw.line([(x1, seg_y), (x1, min(seg_y + 6, y1))], fill=color, width=3)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        r = 7
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(255, 255, 255, 255), width=2)
        edge_tag = "[E] " if is_edge else ""
        text = (f"c{i} {edge_tag}{cand['category']} @{cell}  thr={cand['throat_width_px']}px  "
                f"A={cand['side_a_area_px']}  B={cand['side_b_area_px']}")
        tx = max(2, min(image_size[0] - 440, int(cx) + 12))
        ty = max(2, min(image_size[1] - 22, int(cy) - 20))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=color, font=label_font)

    # Legend in top-right with one row per category, plus an edge-truncated note
    legend_rows = [
        ("DRAIN",        CATEGORY_COLOR["DRAIN"],        "both sides have substantial water"),
        ("CREEK_MOUTH",  CATEGORY_COLOR["CREEK_MOUTH"],  "throat >= 40, water continues inland (low compactness)"),
        ("LARGE_POCKET", CATEGORY_COLOR["LARGE_POCKET"], "throat >= 40, dead-end indent (high compactness)"),
        ("SHOAL",        CATEGORY_COLOR["SHOAL"],        "open-water constriction (20-40 px)"),
    ]
    extra_note = "Dashed box + [E] tag = is_edge_truncated (LLM verifier should re-judge)"
    line_h = 22
    box_w = 540
    box_h = line_h * (len(legend_rows) + 2) + 14
    bx0 = image_size[0] - box_w - 8
    by0 = 8
    draw.rectangle([bx0, by0, bx0 + box_w, by0 + box_h],
                   fill=(0, 0, 0, 210), outline=(255, 255, 255, 255), width=2)
    draw.text((bx0 + 8, by0 + 6),
              f"Phase 1 candidates  (R={SCALE_RADIUS}, kept categories only)",
              fill=(255, 255, 255, 255), font=label_font)
    ty = by0 + 6 + line_h
    for label, color, descr in legend_rows:
        sx = bx0 + 8
        draw.rectangle([sx, ty + 3, sx + 18, ty + 19], fill=color)
        draw.text((sx + 26, ty), f"{label}  —  {descr}",
                  fill=(255, 255, 255, 255), font=label_font)
        ty += line_h
    draw.text((bx0 + 8, ty), extra_note,
              fill=(220, 220, 220, 255), font=label_font)

    Image.alpha_composite(base, layer).convert("RGB").save(output_path)
    return output_path


# ---- CLI ----


def run_one(cell_id: str) -> int:
    parent_num, child_num = cell_id.removeprefix("root-").split("-")
    stem = f"z0_{parent_num}_{child_num}"
    water_mask_path = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2_google_water"
                       / f"{cell_id}_water_mask.png")
    z16_path = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
                / f"{stem}.png")
    out_dir = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
               / "structures" / cell_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not water_mask_path.exists() or not z16_path.exists():
        print(f"{cell_id}: missing inputs. Skipping.")
        return 1

    print(f"--- {cell_id} ---")
    mask_img = Image.open(water_mask_path).convert("L")
    if mask_img.size != (1280, 1280):
        mask_img = mask_img.resize((1280, 1280), Image.NEAREST)
    water = np.array(mask_img) > 127

    print(f"detecting constrictions (R={SCALE_RADIUS})...")
    candidates, wide_mask = detect_constrictions(water)
    edge_count = sum(1 for c in candidates if c.get("is_edge_truncated"))
    print(f"  {len(candidates)} initial kept; {edge_count} edge-truncated")

    if edge_count:
        print(f"  revalidating edge candidates against wide z14 view...")
        n_in, n_ok, n_drop = revalidate_edges_at_z14(candidates, cell_id)
        print(f"  z14 revalidation: {n_in} edge candidates checked, "
              f"{n_ok} confirmed, {n_drop} dropped (no z14 constriction)")

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"  {len(candidates)} final kept: "
          f"DRAIN={counts.get('DRAIN', 0)}  "
          f"CREEK_MOUTH={counts.get('CREEK_MOUTH', 0)}  "
          f"LARGE_POCKET={counts.get('LARGE_POCKET', 0)}  "
          f"SHOAL={counts.get('SHOAL', 0)}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = out_dir / f"cv_phase1_candidates_{ts}.png"
    json_path = out_dir / f"cv_phase1_candidates_{ts}.json"

    render_overlay(z16_path, candidates, wide_mask, overlay_path)
    json_path.write_text(json.dumps({
        "cell_id": cell_id,
        "phase": 1,
        "scale_radius_px": SCALE_RADIUS,
        "candidates": [
            {
                "id": f"c{i}",
                "category": c["category"],
                "is_edge_truncated": c.get("is_edge_truncated", False),
                "confirmed_at_z14": c.get("confirmed_at_z14", False),
                "pixel_bbox": list(c["bbox"]),
                "pixel_center": [round(c["center"][0], 1), round(c["center"][1], 1)],
                "throat_width_px": c["throat_width_px"],
                "narrow_area_px": c["narrow_area_px"],
                "side_a_area_px": c["side_a_area_px"],
                "side_b_area_px": c["side_b_area_px"],
                "n_adjacent_wides": c["n_adjacent_wides"],
                "water_density": c["water_density"],
                "inland_compactness": c["inland_compactness"],
                "scale_radius_px": c.get("scale_radius_px", SCALE_RADIUS),
            }
            for i, c in enumerate(candidates, start=1)
        ],
    }, indent=2), encoding="utf-8")

    if candidates:
        print(f"\n  {'id':<5s} {'cat':<13s} {'cell':<4s} {'edge':>5s} "
              f"{'throat':>6s} {'A_area':>9s} {'B_area':>9s} {'wdens':>5s} {'compact':>7s}")
        for i, c in enumerate(candidates, start=1):
            cell = _grid_cell_for(c["center"][0], c["center"][1])
            if c.get("is_edge_truncated"):
                # E*  = confirmed at z14;  E?  = present (shouldn't happen post-revalidation);
                edge_flag = "E*" if c.get("confirmed_at_z14") else "E?"
            else:
                edge_flag = "  "
            print(f"  c{i:<4d} {c['category']:<13s} {cell:<4s} {edge_flag:>5s} "
                  f"{c['throat_width_px']:>4d}px "
                  f"{c['side_a_area_px']:>9d} {c['side_b_area_px']:>9d} "
                  f"{c['water_density']:>5.2f} {c['inland_compactness']:>7.3f}")
    print(f"\n  overlay: {overlay_path.relative_to(REPO_ROOT)}")
    print(f"  json:    {json_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True,
                        help="Cell id like root-10-8. Repeat for multiple.")
    args = parser.parse_args()
    rc_overall = 0
    for cell_id in args.cell:
        rc = run_one(cell_id)
        if rc != 0:
            rc_overall = rc
    return rc_overall


if __name__ == "__main__":
    sys.exit(main())
