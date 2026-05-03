"""CV-based point candidate detection from the per-cell water mask.

A POINT is a peninsula or promontory — a piece of land that juts out into
water. Detection runs the same wide-narrow split that the drains detector
uses, but on the LAND mask instead of the WATER mask, and at multiple
scales. The narrow_land connected component IS the peninsula body — its
pixel count is the body area.

PIPELINE (per scale R)

  1. land = NOT water (raw, no smoothing — smoothing would close water gaps
     and merge distinct landmasses).
  2. wide_land = open(land, R) — substantial-width land bodies (the mainland).
  3. narrow_land = land - wide_land — strips of land ≤ ~2R px wide.
  4. CC scan on narrow_land. Drop CCs smaller than the scale's min_body_area
     or above MAX_BODY_AREA (= mainland-sized).
  5. Drop CCs whose bbox touches the cell frame (edge artifacts).
  6. Require ≥ 1 adjacent wide_land CC (the mainland the peninsula attaches
     to). Tip = pixel of the narrow_land CC farthest from mainland centroid.
  7. Tag the candidate with the scale label.

MULTI-SCALE

  POINT_R13 — R=13 catches narrow features ≤ ~26 px wide; body ≥ 200 px.
              Picks up small mangrove fingers + tapered tips of larger
              peninsulas where the tip is < 26 px wide.
  POINT_R26 — R=26 catches features ≤ ~52 px wide; body ≥ 400 px.
              Picks up the same R13 features PLUS broader-bodied peninsulas
              (since narrow at R=13 is a subset of narrow at R=26).

  The label names the SCALE that produced the candidate, NOT the physical
  size. A POINT_R26 may have a smaller body area than a POINT_R13 — they're
  catching different geometric patterns. Same physical peninsula can appear
  in both; spatial dedup is a future task.

EDGE HANDLING

  Edge-touching narrow_land CCs are dropped outright. At this scale, edge
  artifacts dominate over real edge-truncated peninsulas — much cleaner to
  drop than to revalidate.

All path construction goes through ``readwater.storage``; mask layout is
``data/areas/<area>/masks/water/`` (see ``readwater.storage`` for details).

Usage (via shim):
  python scripts/cv_detect_points.py --cell root-10-8
  python scripts/cv_detect_points.py --cell root-2-9 --cell root-11-1 --cell root-11-5
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
    bbox_touches_frame,
    connected_components,
    find_adjacent,
    grid_cell_for,
    load_font,
    open_mask,
)

SCHEMA_VERSION = "points.v1"

# ---- Point parameters ----

# Multi-scale detection: each scale catches its own tier of point. Doubling R
# from one tier to the next captures peninsulas with progressively wider
# bodies; doubling min_area requires the bigger-tier candidates to actually
# carry more pixels (otherwise narrower features would just leak through).
#
#   POINT_R13 — R=13, narrow ≤ ~26 px wide, body area ≥ 200 px
#   POINT_R26 — R=26, narrow ≤ ~52 px wide, body area ≥ 400 px
#
# The label names the SCALE that produced the candidate (R-value), not the
# physical size of the point — a R26 candidate may have a smaller body area
# than a R13 one. Higher-R passes catch wider peninsulas (narrow at R=13 is
# a subset of narrow at R=26). The same physical peninsula can appear in
# multiple passes — no spatial dedup yet, accept duplicates for now.
SCALES: list[dict] = [
    {"label": "POINT_R13", "scale_radius": 13, "min_body_area": 200},
    {"label": "POINT_R26", "scale_radius": 26, "min_body_area": 400},
]

MIN_WIDE_LAND_AREA = 200     # wide_land CCs smaller than this aren't valid "mainland"
MAX_BODY_AREA = 100_000      # above = body is mainland-sized, drop regardless of tier


def find_tip_pixel(body_pixels: set[tuple[int, int]],
                   mainland_center: tuple[float, float]) -> tuple[int, int]:
    """Return the pixel in the peninsula body farthest from the mainland
    centroid — the "tip" of the point (most fishable spot).
    """
    mcx, mcy = mainland_center
    best = None
    best_d2 = -1.0
    for (y, x) in body_pixels:
        d2 = (x - mcx) ** 2 + (y - mcy) ** 2
        if d2 > best_d2:
            best_d2 = d2
            best = (x, y)
    return best  # (x, y)


# ---- Detection ----


def _detect_at_scale(water: np.ndarray, scale: dict) -> list[dict]:
    """Run the narrow_land → CC → filter pipeline at a single scale.

    A "candidate" here = one connected component of narrow_land at this scale,
    after the noise / edge / mainland-attachment filters. Tagged with the
    scale's category label (POINT_SMALL, POINT_MEDIUM, ...).
    """
    R = scale["scale_radius"]
    min_area = scale["min_body_area"]
    label = scale["label"]

    land = ~water
    wide_land = open_mask(land, R)
    narrow_land = land & ~wide_land

    narrow_ccs = connected_components(narrow_land, min_pixels=min_area)
    wide_ccs = connected_components(wide_land, min_pixels=MIN_WIDE_LAND_AREA)

    kept: list[dict] = []
    for nc in narrow_ccs:
        if nc["area"] > MAX_BODY_AREA:
            continue   # body is mainland-sized, drop
        # Drop edge-touching outright (no z14 revalidation for points)
        if bbox_touches_frame(nc["bbox"], water.shape, EDGE_MARGIN_PX):
            continue
        # Must be attached to at least one mainland
        adj = find_adjacent(nc["pixels"], wide_ccs, water.shape)
        if len(adj) < 1:
            continue
        adj.sort(key=lambda c: c["area"], reverse=True)
        mainland = adj[0]

        bbox = nc["bbox"]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        neck_width = min(bw, bh)
        tip = find_tip_pixel(nc["pixels"], mainland["center"])

        kept.append({
            "bbox": bbox,
            "center": nc["center"],
            "tip": tip,
            "neck_center": nc["center"],
            "neck_width_px": neck_width,
            "neck_area_px": nc["area"],
            "body_area_px": nc["area"],
            "mainland_area_px": mainland["area"],
            "n_adjacent_wides": len(adj),
            "category": label,
            "scale_radius_px": R,
        })
    return kept


def detect_points(water: np.ndarray) -> list[dict]:
    """Run every scale in SCALES and concatenate the kept candidates.

    Walk-the-narrow-inland approach, multi-scale:

      For each (scale_radius R, min_body_area):
        1. land = ~water; wide_land = open(land, R); narrow_land = land - wide_land
        2. CC scan on narrow_land, drop CCs smaller than min_body_area
        3. Drop anything whose bbox touches the cell frame
        4. Drop CCs above MAX_BODY_AREA (mainland-sized, not a "point")
        5. Require at least 1 adjacent wide_land (the mainland it attaches to)
        6. Tip = pixel of CC farthest from mainland centroid

    Higher-R passes catch wider peninsulas; doubling min_area in lockstep
    keeps higher tiers from collecting noise. The same physical peninsula
    can appear in multiple passes at different sizes — no dedup yet.
    """
    all_kept: list[dict] = []
    for scale in SCALES:
        all_kept.extend(_detect_at_scale(water, scale))
    return all_kept


# ---- Rendering ----

CATEGORY_COLOR = {
    "POINT_R13": (255, 150, 200, 255),    # pink — caught at R=13 (narrow ≤ ~26 px)
    "POINT_R26": (220,  80, 180, 255),    # magenta — caught at R=26 (narrow ≤ ~52 px)
}
LAND_TINT = (180, 140, 80, 50)               # soft tan tint to show land context


def render_overlay(base_image_path: Path,
                   candidates: list[dict],
                   land_mask: np.ndarray,
                   output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Tint the land mask for context
    h, w = land_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[land_mask] = LAND_TINT
    land_layer = Image.fromarray(rgba, mode="RGBA")
    layer = Image.alpha_composite(layer, land_layer)
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
        tip_x, tip_y = cand["tip"]
        neck_cx, neck_cy = cand["neck_center"]
        cell = grid_cell_for(cx, cy, image_size)
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        x0 = max(0, bbox[0] - 2)
        y0 = max(0, bbox[1] - 2)
        x1 = min(image_size[0], bbox[2] + 2)
        y1 = min(image_size[1], bbox[3] + 2)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Connect neck → tip with a thin line so the peninsula "axis" is visible
        draw.line([(int(neck_cx), int(neck_cy)), (int(tip_x), int(tip_y))],
                  fill=color, width=2)

        # Tip = big circle (the fishable spot)
        tip_r = 8
        draw.ellipse([tip_x - tip_r, tip_y - tip_r, tip_x + tip_r, tip_y + tip_r],
                     fill=color, outline=(255, 255, 255, 255), width=2)

        # Neck centroid = small X marker (where the peninsula attaches)
        neck_cx_i = int(neck_cx); neck_cy_i = int(neck_cy)
        x_size = 5
        draw.line([(neck_cx_i - x_size, neck_cy_i - x_size),
                   (neck_cx_i + x_size, neck_cy_i + x_size)], fill=color, width=2)
        draw.line([(neck_cx_i - x_size, neck_cy_i + x_size),
                   (neck_cx_i + x_size, neck_cy_i - x_size)], fill=color, width=2)

        text = (f"p{i} {cand['category']} @{cell}  "
                f"body={cand['body_area_px']}px  neck={cand['neck_width_px']}px")
        tx = max(2, min(image_size[0] - 380, int(tip_x) + 12))
        ty = max(2, min(image_size[1] - 22, int(tip_y) - 18))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=color, font=label_font)

    # Legend (top-right)
    legend_rows = [
        ("POINT_R13", CATEGORY_COLOR["POINT_R13"],
         f"caught at R=13 (narrow ≤ ~26 px), body ≥ {SCALES[0]['min_body_area']} px"),
        ("POINT_R26", CATEGORY_COLOR["POINT_R26"],
         f"caught at R=26 (narrow ≤ ~52 px), body ≥ {SCALES[1]['min_body_area']} px"),
    ]
    extra_lines = [
        "Big circle = TIP (fishable spot)",
        "X = neck (where peninsula attaches to mainland)",
        "Edge-touching candidates are dropped, not flagged.",
    ]
    line_h = 22
    box_w = 580
    box_h = line_h * (len(legend_rows) + 1 + len(extra_lines)) + 14
    bx0 = image_size[0] - box_w - 8
    by0 = 8
    draw.rectangle([bx0, by0, bx0 + box_w, by0 + box_h],
                   fill=(0, 0, 0, 210), outline=(255, 255, 255, 255), width=2)
    draw.text((bx0 + 8, by0 + 6),
              "Points  (multi-scale, narrow_land CC = body)",
              fill=(255, 255, 255, 255), font=label_font)
    ty = by0 + 6 + line_h
    for label, color, descr in legend_rows:
        sx = bx0 + 8
        draw.rectangle([sx, ty + 3, sx + 18, ty + 19], fill=color)
        draw.text((sx + 26, ty), f"{label}  —  {descr}",
                  fill=(255, 255, 255, 255), font=label_font)
        ty += line_h
    for line in extra_lines:
        draw.text((bx0 + 8, ty), line, fill=(220, 220, 220, 255), font=label_font)
        ty += line_h

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

    scale_summary = ", ".join(
        f"R={s['scale_radius']} min={s['min_body_area']}" for s in SCALES
    )
    print(f"detecting points multi-scale ({scale_summary}; edges dropped)...")
    candidates = detect_points(water)

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    parts = [f"{lbl.removeprefix('POINT_')}={n}"
             for lbl, n in sorted(counts.items())]
    print(f"  {len(candidates)} final kept: " + "  ".join(parts))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = storage.cell_artifact_path(area_id, cell_id, "points", ts, "png")
    json_path = storage.cell_artifact_path(area_id, cell_id, "points", ts, "json")

    land_mask = ~water
    render_overlay(z16_path, candidates, land_mask, overlay_path)

    storage.atomic_write_json(json_path, {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "detector": "cv_detect_points",
        "scales": [{"label": s["label"], "scale_radius": s["scale_radius"],
                    "min_body_area": s["min_body_area"]} for s in SCALES],
        "candidates": [
            {
                "id": f"p{i}",
                "category": c["category"],
                "scale_radius_px": c.get("scale_radius_px"),
                "pixel_bbox": list(c["bbox"]),
                "pixel_center": [round(c["center"][0], 1), round(c["center"][1], 1)],
                "pixel_tip": list(c["tip"]),
                "neck_center": [round(c["neck_center"][0], 1), round(c["neck_center"][1], 1)],
                "neck_width_px": c["neck_width_px"],
                "neck_area_px": c["neck_area_px"],
                "body_area_px": c["body_area_px"],
                "mainland_area_px": c["mainland_area_px"],
                "n_adjacent_wides": c["n_adjacent_wides"],
            }
            for i, c in enumerate(candidates, start=1)
        ],
    })

    if candidates:
        print(f"\n  {'id':<5s} {'cat':<13s} {'cell':<4s} {'R':>3s} "
              f"{'body':>7s} {'neck':>5s} {'mainland':>9s}")
        for i, c in enumerate(candidates, start=1):
            cell = grid_cell_for(c["center"][0], c["center"][1])
            R = c.get("scale_radius_px", "")
            print(f"  p{i:<4d} {c['category']:<13s} {cell:<4s} {R!s:>3s} "
                  f"{c['body_area_px']:>5d}px {c['neck_width_px']:>3d}px "
                  f"{c['mainland_area_px']:>8d}px")
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
