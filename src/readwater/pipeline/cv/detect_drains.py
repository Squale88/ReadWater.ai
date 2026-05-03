"""CV-based drain / creek-mouth / large-pocket / shoal candidate detection.

Operates on the per-cell Google z16 water mask. Finds narrow water strips
("constrictions") at scale R=25 (catches throats up to ~50 px wide) and
keeps only those that match one of four rules:

  DRAIN         — both adjacent wide-water sides are substantial
                  (side_b >= DRAIN_MIN_SIDE_B_AREA)
  CREEK_MOUTH   — one big bay, throat >= 40 px, water continues inland
                  (low inland-water compactness — long thin shape)
  LARGE_POCKET  — one big bay, throat >= 40 px, dead-end indent
                  (high inland-water compactness — roundish shape)
  SHOAL         — one big bay, throat 20-40 px, surrounded by water
                  (high water density in candidate neighborhood)

Anything else (small mangrove pockets, interior creek sections, edge artifacts,
mask noise) is dropped. Edge candidates whose throats sit within EDGE_MARGIN_PX
of the cell frame are revalidated against the wide z14 mask: if the same
constriction doesn't reproduce at the wider scale, the candidate was a framing
artifact and gets dropped. Otherwise it's kept and marked confirmed_at_z14.

The downstream LLM verifier owns the final call on each kept candidate; the
rules here just keep the input set focused on features likely to be anchors.

Algorithm:
  1. Smooth the water mask (boundary cleanup).
  2. wide = open(water, R). narrow = water - wide.
  3. Connected components on narrow (>= MIN_NARROW_AREA) and wide (>= MIN_WIDE_AREA).
  4. For each narrow CC: measure throat width, two largest adjacent wide
     areas, water density in neighborhood, and (for >=40px throats) inland
     compactness.
  5. classify() picks one of the four categories or drops the candidate.
  6. revalidate_edges_at_z14() drops edge artifacts.

Render kept candidates color-coded by category on the satellite.

All path construction goes through ``readwater.storage``; mask layout is
``data/areas/<area>/masks/water/`` (see ``readwater.storage`` for details).

Usage (via shim):
  python scripts/cv_detect_drains.py --cell root-10-8
  python scripts/cv_detect_drains.py --cell root-2-9 --cell root-11-1 --cell root-11-5
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from readwater import storage
from readwater.pipeline.cv.helpers import (
    EDGE_MARGIN_PX,
    SMOOTH_RADIUS,
    bbox_touches_frame,
    connected_components,
    find_adjacent,
    grid_cell_for,
    load_font,
    load_z14_water_mask,
    open_mask,
    smooth_mask,
    water_density_around,
    z16_to_z14,
)

SCHEMA_VERSION = "drains.v1"

# ---- Drain-detector parameters ----

SCALE_RADIUS = 25          # captures throats up to ~50 px wide (medium scale)
MIN_NARROW_AREA = 60       # drop narrow CCs smaller than this (mask-noise floor)
MIN_WIDE_AREA = 200        # drop wide CCs smaller than this when looking for "sides"

# 4-category classification thresholds (derived from manual classification of
# root-11-5; revisit on other cells when adding more test data).
DRAIN_MIN_SIDE_B_AREA = 500       # second-side area required to call a feature a DRAIN
MOUTH_MIN_SIDE_A_AREA = 5_000     # first-side area required to call ANY single-side feature
MOUTH_MIN_THROAT_PX = 40          # throat width to qualify as a CREEK MOUTH or LARGE_POCKET
SHOAL_MIN_THROAT_PX = 20          # lower bound for SHOAL throat (anything below is too small)
SHOAL_MAX_THROAT_PX = 40          # upper bound (above goes into MOUTH/POCKET category)
SHOAL_NEIGHBORHOOD_RADIUS = 100   # px radius around the candidate to measure water density
SHOAL_MIN_WATER_DENSITY = 0.65    # neighborhood >= this fraction water -> SHOAL

# CREEK_MOUTH vs LARGE_POCKET: both have one substantial wide side and a
# throat >= 40 px. The split is by the SHAPE of the inland water reachable
# from the throat after subtracting side_a:
#   - creek mouth: water snakes inland (low compactness — long thin shape)
#   - large pocket: water fills a roundish bay (high compactness)
# Threshold tuned on root-11-5 (c15 = 0.302 vs pockets 0.37-0.59).
CREEK_MOUTH_MAX_INLAND_COMPACTNESS = 0.35

# Edge revalidation against z14
EDGE_REVALIDATE_MATCH_RADIUS_Z14 = 25   # z14 px (~100 z16 ground px)
EDGE_REVALIDATE_MIN_NARROW_AREA_Z14 = 6 # min narrow CC pixels at z14 (1/16 area ratio)


# ---- Drain-specific measurements ----


def inland_compactness(smoothed_water: np.ndarray,
                       narrow_pixels: set[tuple[int, int]],
                       side_a_pixels: set[tuple[int, int]]) -> float:
    """Compactness (area / bbox_area) of water reachable from the throat after
    subtracting side_a's main wide CC.

    A CREEK MOUTH has water that snakes inland through mangrove (long thin
    shape -> low compactness). A LARGE POCKET has water that fills a roundish
    bay (compact shape -> high compactness).

    Returns 0.0 when there's no measurable inland water (the seed pixel is
    inside side_a). Callers should treat 0.0 as "couldn't measure" rather
    than "very elongated."
    """
    h, w = smoothed_water.shape
    side_a_mask = np.zeros((h, w), dtype=bool)
    for (y, x) in side_a_pixels:
        side_a_mask[y, x] = True
    inland = smoothed_water & ~side_a_mask

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


# ---- Classification ----


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
        if compactness < CREEK_MOUTH_MAX_INLAND_COMPACTNESS:
            return "CREEK_MOUTH"
        return "LARGE_POCKET"
    if SHOAL_MIN_THROAT_PX <= throat < MOUTH_MIN_THROAT_PX:
        if density >= SHOAL_MIN_WATER_DENSITY:
            return "SHOAL"
    return None


# ---- Detection ----


def detect_constrictions(water: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Single-scale detection at SCALE_RADIUS. Returns (kept_candidates, wide_mask).

    Each kept candidate dict carries: bbox, center, narrow_area_px,
    throat_width_px, side_a_area_px, side_b_area_px, n_adjacent_wides,
    water_density, inland_compactness, scale_radius_px, is_edge_truncated,
    category.
    """
    smoothed = smooth_mask(water, radius=SMOOTH_RADIUS)
    wide = open_mask(smoothed, SCALE_RADIUS)
    narrow = smoothed & ~wide

    narrow_ccs = connected_components(narrow, min_pixels=MIN_NARROW_AREA)
    wide_ccs = connected_components(wide, min_pixels=MIN_WIDE_AREA)

    kept: list[dict] = []
    for nc in narrow_ccs:
        bbox = nc["bbox"]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        throat_width = min(bw, bh)

        adj = find_adjacent(nc["pixels"], wide_ccs, water.shape)
        adj.sort(key=lambda c: c["area"], reverse=True)
        side_a_cc = adj[0] if len(adj) >= 1 else None
        side_a = side_a_cc["area"] if side_a_cc else 0
        side_b = adj[1]["area"] if len(adj) >= 2 else 0

        density = water_density_around(smoothed, nc["center"], SHOAL_NEIGHBORHOOD_RADIUS)

        # Compactness only matters for the >=40px creek/pocket split.
        if (side_a_cc is not None
                and throat_width >= MOUTH_MIN_THROAT_PX
                and side_b < DRAIN_MIN_SIDE_B_AREA):
            compactness = inland_compactness(smoothed, nc["pixels"], side_a_cc["pixels"])
        else:
            compactness = 0.0

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
            "scale_radius_px": SCALE_RADIUS,
            "is_edge_truncated": bbox_touches_frame(bbox, water.shape, EDGE_MARGIN_PX),
        }
        cat = classify(cand)
        if cat is None:
            continue
        cand["category"] = cat
        kept.append(cand)

    return kept, wide


# ---- Edge revalidation against z14 ----


def revalidate_edges_at_z14(candidates: list[dict],
                            area_id: str,
                            cell_id: str) -> tuple[int, int, int]:
    """For each is_edge_truncated candidate, re-run the wide/narrow split on
    the wider z14 mask and DROP those that don't have a matching narrow CC
    near the edge candidate's z14-mapped location.

    Insight: most z16 edge "features" aren't really features — they're
    artifacts of the cell boundary cutting through what's really open water,
    a continuous shoreline, or a mangrove edge. When the wider context is
    visible at z14, those artifacts disappear. The ones that survive are
    real features whose throat happens to fall near the cell border.

    Augments surviving candidates with confirmed_at_z14: True.

    Returns (n_edge_input, n_confirmed, n_dropped). No-op if the wide z14
    styled image isn't on disk.
    """
    edge_indices = [i for i, c in enumerate(candidates) if c.get("is_edge_truncated")]
    if not edge_indices:
        return (0, 0, 0)

    z14_water = load_z14_water_mask(area_id, cell_id)
    if z14_water is None:
        print(f"  no z14 wide tile for {cell_id} — skipping edge revalidation")
        return (0, 0, 0)

    # Same wide/narrow split at z14 with R scaled to match z16 ground scale
    # (1 z14 pixel = 4 z16 ground pixels, so z14_R = z16_R / 4).
    smoothed_z14 = smooth_mask(z14_water, radius=SMOOTH_RADIUS)
    z14_R = max(1, SCALE_RADIUS // 4)            # = 6
    wide_z14 = open_mask(smoothed_z14, z14_R)
    narrow_z14 = smoothed_z14 & ~wide_z14
    z14_narrow_ccs = connected_components(
        narrow_z14, min_pixels=EDGE_REVALIDATE_MIN_NARROW_AREA_Z14,
    )

    drop_indices: list[int] = []
    n_confirmed = 0
    match_dist_sq = EDGE_REVALIDATE_MATCH_RADIUS_Z14 ** 2

    for i in edge_indices:
        c = candidates[i]
        z14_cx, z14_cy = z16_to_z14(c["center"][0], c["center"][1])
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


CATEGORY_COLOR = {
    "DRAIN":        (220,  30,  30, 255),    # red
    "CREEK_MOUTH":  (255, 165,   0, 255),    # orange — rare, true creek extending inland
    "LARGE_POCKET": (255, 230,  50, 255),    # yellow — common, dead-end indent
    "SHOAL":        ( 60, 200, 230, 255),    # cyan-ish — open-water constriction
}
WIDE_TINT = (80, 170, 220, 60)


def render_overlay(base_image_path: Path,
                   candidates: list[dict],
                   wide_mask: np.ndarray,
                   output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Tint wide-water bodies for context.
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
    grid_font = load_font(max(10, int(min(cell_w, cell_h) * 0.30)))
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

    label_font = load_font(15)
    for i, cand in enumerate(candidates, start=1):
        bbox = cand["bbox"]
        cx, cy = cand["center"]
        cell = grid_cell_for(cx, cy, image_size)
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        x0 = max(0, bbox[0] - 4)
        y0 = max(0, bbox[1] - 4)
        x1 = min(image_size[0], bbox[2] + 4)
        y1 = min(image_size[1], bbox[3] + 4)
        is_edge = cand.get("is_edge_truncated", False)
        if is_edge:
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

    # Legend (top-right) with one row per category + edge-truncated note
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
              f"Drains  (R={SCALE_RADIUS}, kept categories only)",
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

    composed = Image.alpha_composite(base, layer).convert("RGB")
    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    storage.atomic_write_bytes(output_path, buf.getvalue())
    return output_path


# ---- CLI ----


def run_one(area_id: str, cell_id: str) -> int:
    water_mask_path = storage.water_mask_path(area_id, cell_id)
    z16_path = storage.z16_image_path(area_id, cell_id)
    out_dir = storage.cell_structures_dir(area_id, cell_id)
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
        n_in, n_ok, n_drop = revalidate_edges_at_z14(candidates, area_id, cell_id)
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
    overlay_path = storage.cell_artifact_path(area_id, cell_id, "drains", ts, "png")
    json_path = storage.cell_artifact_path(area_id, cell_id, "drains", ts, "json")

    render_overlay(z16_path, candidates, wide_mask, overlay_path)
    storage.atomic_write_json(json_path, {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "detector": "cv_detect_drains",
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
    })

    if candidates:
        print(f"\n  {'id':<5s} {'cat':<13s} {'cell':<4s} {'edge':>5s} "
              f"{'throat':>6s} {'A_area':>9s} {'B_area':>9s} {'wdens':>5s} {'compact':>7s}")
        for i, c in enumerate(candidates, start=1):
            cell = grid_cell_for(c["center"][0], c["center"][1])
            if c.get("is_edge_truncated"):
                edge_flag = "E*" if c.get("confirmed_at_z14") else "E?"
            else:
                edge_flag = "  "
            print(f"  c{i:<4d} {c['category']:<13s} {cell:<4s} {edge_flag:>5s} "
                  f"{c['throat_width_px']:>4d}px "
                  f"{c['side_a_area_px']:>9d} {c['side_b_area_px']:>9d} "
                  f"{c['water_density']:>5.2f} {c['inland_compactness']:>7.3f}")
    print(f"\n  overlay: {overlay_path.relative_to(storage.data_root().parent)}")
    print(f"  json:    {json_path.relative_to(storage.data_root().parent)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="rookery_bay_v2",
                        help="Area id (default: rookery_bay_v2).")
    parser.add_argument("--cell", action="append", required=True,
                        help="Cell id like root-10-8. Repeat for multiple.")
    args = parser.parse_args()
    rc_overall = 0
    for cell_id in args.cell:
        rc = run_one(args.area, cell_id)
        if rc != 0:
            rc_overall = rc
    return rc_overall


if __name__ == "__main__":
    sys.exit(main())
