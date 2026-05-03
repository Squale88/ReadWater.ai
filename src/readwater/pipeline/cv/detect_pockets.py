"""CV-based small/medium pocket candidate detection from the per-cell water mask.

A POCKET is a small water inlet — the dual of a POINT. Points are narrow
LAND extending into water; pockets are narrow WATER extending into land.
Same algorithm as cv_detect_points.py, just applied to the water mask
directly (no inversion).

The cv_detect_drains.py classifier already labels LARGE_POCKET (throat
≥ 40 px, high inland_compactness). This detector covers the SMALLER end
that drains doesn't bother with — small back-water pockets, narrow inlets,
and tiny coves that are good fishing spots but don't fit the drain rules.

PIPELINE (per scale R)

  1. wide_water = open(water, R) — substantial-width water bodies (the
     bay / main water the pocket opens into).
  2. narrow_water = water - wide_water — strips of water ≤ ~2R px wide.
  3. CC scan on narrow_water. Each narrow_water CC IS the pocket body —
     its pixel count is the body area.
  4. Drop CCs smaller than the scale's min_body_area or above MAX_BODY_AREA
     (pocket-sized features only; mainland-water-sized = drop).
  5. Drop CCs whose bbox touches the cell frame (edge artifacts).
  6. Require ≥ 1 adjacent wide_water CC (the bay the pocket opens into).
  7. Tip (back-of-pocket) = pixel of narrow_water CC farthest from the
     bay's centroid — that's the "deep end" of the pocket.

MULTI-SCALE

  POCKET_R13 — R=13 (narrow ≤ ~26 px wide), body ≥ 200 px
               Catches small back-water pockets / narrow inlets.
  POCKET_R26 — R=26 (narrow ≤ ~52 px wide), body ≥ 400 px
               Catches everything R13 catches PLUS broader-bodied pockets
               (since narrow at R=13 is a subset of narrow at R=26).

  The label names the SCALE that produced the candidate, NOT the physical
  size of the pocket. A POCKET_R26 may have a smaller body area than a
  POCKET_R13. Same physical pocket can appear in both passes; spatial
  dedup is a future task.

OVERLAP WITH DRAINS

  This detector intentionally overlaps with cv_detect_drains.py. The drain
  classifier catches LARGE_POCKETs (throat ≥ 40 px); this catches the
  smaller-throat pockets that drain doesn't surface. Same physical feature
  may appear in BOTH detectors' output. The orchestrator (Step 5) is the
  place to resolve overlap — likely by preferring the most informative
  category (DRAIN > LARGE_POCKET > POCKET_R26 > POCKET_R13).

EDGE HANDLING

  Edge-touching narrow_water CCs are dropped outright. Same lesson as the
  points detector: at this scale, edge artifacts dominate over real
  edge-truncated pockets.

All path construction goes through ``readwater.storage``; mask layout is
``data/areas/<area>/masks/water/`` (see ``readwater.storage`` for details).

Usage (via shim):
  python scripts/cv_detect_pockets.py --cell root-10-8
  python scripts/cv_detect_pockets.py --cell root-2-9 --cell root-11-1 --cell root-11-5
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

SCHEMA_VERSION = "pockets.v1"

# ---- Pocket parameters ----

# Multi-scale: each scale catches its own tier of pocket. Doubling R from
# one tier to the next captures pockets with progressively wider bodies;
# doubling min_area in lockstep keeps higher tiers from collecting noise.
SCALES: list[dict] = [
    {"label": "POCKET_R13", "scale_radius": 13, "min_body_area": 200},
    {"label": "POCKET_R26", "scale_radius": 26, "min_body_area": 400},
]

MIN_WIDE_WATER_AREA = 200    # wide_water CCs smaller than this aren't valid "main water"
MAX_BODY_AREA = 100_000      # above = body is bay-sized, drop


def find_tip_pixel(body_pixels: set[tuple[int, int]],
                   bay_center: tuple[float, float]) -> tuple[int, int]:
    """Return the pixel in the pocket body farthest from the bay (main water)
    centroid — the back/deep end of the pocket (analog of a peninsula tip).
    """
    bcx, bcy = bay_center
    best = None
    best_d2 = -1.0
    for (y, x) in body_pixels:
        d2 = (x - bcx) ** 2 + (y - bcy) ** 2
        if d2 > best_d2:
            best_d2 = d2
            best = (x, y)
    return best  # (x, y)


# ---- Detection ----


def _detect_at_scale(water: np.ndarray, scale: dict) -> list[dict]:
    """Run the narrow_water → CC → filter pipeline at a single scale.

    A "candidate" here = one connected component of narrow_water at this scale,
    after the noise / edge / bay-attachment filters. Tagged with the scale's
    category label (POCKET_R13, POCKET_R26).
    """
    R = scale["scale_radius"]
    min_area = scale["min_body_area"]
    label = scale["label"]

    wide_water = open_mask(water, R)
    narrow_water = water & ~wide_water

    narrow_ccs = connected_components(narrow_water, min_pixels=min_area)
    wide_ccs = connected_components(wide_water, min_pixels=MIN_WIDE_WATER_AREA)

    kept: list[dict] = []
    for nc in narrow_ccs:
        if nc["area"] > MAX_BODY_AREA:
            continue   # body is bay-sized, drop
        if bbox_touches_frame(nc["bbox"], water.shape, EDGE_MARGIN_PX):
            continue
        # Must be attached to at least one main water body
        adj = find_adjacent(nc["pixels"], wide_ccs, water.shape)
        if len(adj) < 1:
            continue
        adj.sort(key=lambda c: c["area"], reverse=True)
        bay = adj[0]

        bbox = nc["bbox"]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        # "Mouth width" = short bbox dim of the pocket (analog of neck width
        # for points). Approximation: assumes the pocket is roughly elongated.
        mouth_width = min(bw, bh)
        # "Tip" = pixel of narrow_water CC farthest from the bay centroid =
        # the back/deep end of the pocket.
        tip = find_tip_pixel(nc["pixels"], bay["center"])

        kept.append({
            "bbox": bbox,
            "center": nc["center"],
            "tip": tip,
            "mouth_center": nc["center"],
            "mouth_width_px": mouth_width,
            "mouth_area_px": nc["area"],
            "body_area_px": nc["area"],
            "bay_area_px": bay["area"],
            "n_adjacent_wides": len(adj),
            "category": label,
            "scale_radius_px": R,
        })
    return kept


def detect_pockets(water: np.ndarray) -> list[dict]:
    """Run every scale in SCALES and concatenate the kept candidates.

    Same multi-scale walk-the-narrow-inland approach as cv_detect_points.py,
    just applied to the water mask directly. Each narrow_water CC IS the
    pocket body — its pixel count is the body area.
    """
    all_kept: list[dict] = []
    for scale in SCALES:
        all_kept.extend(_detect_at_scale(water, scale))
    return all_kept


# ---- Rendering ----

CATEGORY_COLOR = {
    "POCKET_R13": (140, 200, 255, 255),   # light blue — caught at R=13 (narrow ≤ ~26 px)
    "POCKET_R26": ( 30, 110, 200, 255),   # deep blue — caught at R=26 (narrow ≤ ~52 px)
}
WIDE_TINT = (80, 170, 220, 60)            # soft cyan tint over wide-water bodies


def render_overlay(base_image_path: Path,
                   candidates: list[dict],
                   wide_mask: np.ndarray,
                   output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Tint the wide-water mask for context (same as drains overlay)
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
        tip_x, tip_y = cand["tip"]
        mouth_cx, mouth_cy = cand["mouth_center"]
        cell = grid_cell_for(cx, cy, image_size)
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        x0 = max(0, bbox[0] - 2)
        y0 = max(0, bbox[1] - 2)
        x1 = min(image_size[0], bbox[2] + 2)
        y1 = min(image_size[1], bbox[3] + 2)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Connect mouth → tip with a thin line so the pocket "axis" is visible
        draw.line([(int(mouth_cx), int(mouth_cy)), (int(tip_x), int(tip_y))],
                  fill=color, width=2)

        # Tip = big circle (the back/deep end of the pocket — the holding spot)
        tip_r = 8
        draw.ellipse([tip_x - tip_r, tip_y - tip_r, tip_x + tip_r, tip_y + tip_r],
                     fill=color, outline=(255, 255, 255, 255), width=2)

        # Mouth centroid = small X marker (where pocket opens to the bay)
        mcx_i = int(mouth_cx); mcy_i = int(mouth_cy)
        x_size = 5
        draw.line([(mcx_i - x_size, mcy_i - x_size),
                   (mcx_i + x_size, mcy_i + x_size)], fill=color, width=2)
        draw.line([(mcx_i - x_size, mcy_i + x_size),
                   (mcx_i + x_size, mcy_i - x_size)], fill=color, width=2)

        text = (f"k{i} {cand['category']} @{cell}  "
                f"body={cand['body_area_px']}px  mouth={cand['mouth_width_px']}px")
        tx = max(2, min(image_size[0] - 380, int(tip_x) + 12))
        ty = max(2, min(image_size[1] - 22, int(tip_y) - 18))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=color, font=label_font)

    # Legend (top-right)
    legend_rows = [
        ("POCKET_R13", CATEGORY_COLOR["POCKET_R13"],
         f"caught at R=13 (narrow ≤ ~26 px), body ≥ {SCALES[0]['min_body_area']} px"),
        ("POCKET_R26", CATEGORY_COLOR["POCKET_R26"],
         f"caught at R=26 (narrow ≤ ~52 px), body ≥ {SCALES[1]['min_body_area']} px"),
    ]
    extra_lines = [
        "Big circle = TIP (deep end of pocket / back-water spot)",
        "X = mouth (where pocket opens to bay)",
        "Edge-touching candidates are dropped, not flagged.",
    ]
    line_h = 22
    box_w = 600
    box_h = line_h * (len(legend_rows) + 1 + len(extra_lines)) + 14
    bx0 = image_size[0] - box_w - 8
    by0 = 8
    draw.rectangle([bx0, by0, bx0 + box_w, by0 + box_h],
                   fill=(0, 0, 0, 210), outline=(255, 255, 255, 255), width=2)
    draw.text((bx0 + 8, by0 + 6),
              "Pockets  (multi-scale, narrow_water CC = body)",
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
    print(f"detecting pockets multi-scale ({scale_summary}; edges dropped)...")
    candidates = detect_pockets(water)

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    parts = [f"{lbl.removeprefix('POCKET_')}={n}"
             for lbl, n in sorted(counts.items())]
    print(f"  {len(candidates)} final kept: " + "  ".join(parts))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = storage.cell_artifact_path(area_id, cell_id, "pockets", ts, "png")
    json_path = storage.cell_artifact_path(area_id, cell_id, "pockets", ts, "json")

    # Render: tint the R=13 wide-water layer (matches the smaller-scale
    # context; either scale is fine for the cyan background).
    wide_for_render = open_mask(water, SCALES[0]["scale_radius"])
    render_overlay(z16_path, candidates, wide_for_render, overlay_path)

    storage.atomic_write_json(json_path, {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "detector": "cv_detect_pockets",
        "scales": [{"label": s["label"], "scale_radius": s["scale_radius"],
                    "min_body_area": s["min_body_area"]} for s in SCALES],
        "candidates": [
            {
                "id": f"k{i}",
                "category": c["category"],
                "scale_radius_px": c.get("scale_radius_px"),
                "pixel_bbox": list(c["bbox"]),
                "pixel_center": [round(c["center"][0], 1), round(c["center"][1], 1)],
                "pixel_tip": list(c["tip"]),
                "mouth_center": [round(c["mouth_center"][0], 1), round(c["mouth_center"][1], 1)],
                "mouth_width_px": c["mouth_width_px"],
                "mouth_area_px": c["mouth_area_px"],
                "body_area_px": c["body_area_px"],
                "bay_area_px": c["bay_area_px"],
                "n_adjacent_wides": c["n_adjacent_wides"],
            }
            for i, c in enumerate(candidates, start=1)
        ],
    })

    if candidates:
        print(f"\n  {'id':<5s} {'cat':<13s} {'cell':<4s} {'R':>3s} "
              f"{'body':>7s} {'mouth':>6s} {'bay':>9s}")
        for i, c in enumerate(candidates, start=1):
            cell = grid_cell_for(c["center"][0], c["center"][1])
            R = c.get("scale_radius_px", "")
            print(f"  k{i:<4d} {c['category']:<13s} {cell:<4s} {R!s:>3s} "
                  f"{c['body_area_px']:>5d}px {c['mouth_width_px']:>4d}px "
                  f"{c['bay_area_px']:>8d}px")
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
