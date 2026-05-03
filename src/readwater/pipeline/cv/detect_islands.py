"""CV-based island candidate detection from the per-cell water mask.

An "island" is a piece of land surrounded by water on all sides. Geometrically
it's a connected component of `land = NOT water` that doesn't touch the cell
frame (since touching the frame would mean the land continues off-frame).

Three size tiers (z16 pixels ≈ square meters at this latitude):

  ISLAND_SMALL    100 -   2,000 px   ~10–45 m diameter   (mangrove islets)
  ISLAND_MEDIUM  2,000 -  20,000 px  ~45–140 m diameter  (typical mangrove islands)
  ISLAND_LARGE  20,000 - 200,000 px  ~140–450 m diameter (substantial bay islands)

Anything smaller than 100 px is mask noise; anything larger than 200,000 px
is more likely mainland than a discrete island.

EDGE HANDLING

Land CCs whose bbox sits within EDGE_MARGIN_PX of the cell frame are flagged
is_edge_truncated and revalidated against the wide z14 mask. At z14:
  - find the land CC containing the same physical location
  - if THAT z14 land CC also doesn't touch the z14 frame -> still bounded by
    water in the wider view -> real island -> keep, confirmed_at_z14=True
  - if the z14 land CC touches the z14 frame -> off-frame land continues ->
    it's a peninsula, not an island -> drop

All path construction goes through ``readwater.storage``; mask layout is
``data/areas/<area>/masks/water/`` (see ``readwater.storage`` for details).

Usage (via shim):
  python scripts/cv_detect_islands.py --cell root-10-8
  python scripts/cv_detect_islands.py --cell root-2-9 --cell root-11-1 --cell root-11-5
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
    grid_cell_for,
    load_font,
    load_z14_water_mask,
    z16_to_z14,
)

SCHEMA_VERSION = "islands.v1"

# ---- Island parameters ----

MIN_ISLAND_AREA_PX = 100        # below = mask noise
MAX_ISLAND_AREA_PX = 200_000    # above = likely mainland, drop

# Size tier boundaries (area thresholds)
SMALL_MAX = 2_000        # SMALL: 100  - 2k
MEDIUM_MAX = 20_000      # MEDIUM: 2k - 20k
                         # LARGE: 20k - 200k


def classify_island(area_px: int) -> str | None:
    """Return ISLAND_SMALL / ISLAND_MEDIUM / ISLAND_LARGE or None."""
    if area_px < MIN_ISLAND_AREA_PX or area_px > MAX_ISLAND_AREA_PX:
        return None
    if area_px < SMALL_MAX:
        return "ISLAND_SMALL"
    if area_px < MEDIUM_MAX:
        return "ISLAND_MEDIUM"
    return "ISLAND_LARGE"


# ---- Detection ----


def detect_islands(water: np.ndarray) -> list[dict]:
    """Return all kept island candidates from a single z16 water mask.

    Each candidate dict carries:
      bbox, center, area_px, category, is_edge_truncated.

    Important: we do NOT smooth the water mask before inverting to land.
    Smoothing's closing step (dilate-then-erode) fills small water gaps,
    which glues distinct land features (e.g. an island and the mainland
    separated by a thin channel) into one connected component. The drain
    detector needs smoothing to suppress boundary noise that would create
    false narrow-water constrictions, but island detection is the opposite
    problem: we want the natural water gaps preserved so each landmass
    stays its own CC. The MIN_ISLAND_AREA_PX filter handles any speckle
    noise the absence of smoothing might let through.
    """
    land = ~water
    land_ccs = connected_components(land, min_pixels=MIN_ISLAND_AREA_PX)

    kept: list[dict] = []
    for cc in land_ccs:
        area = cc["area"]
        cat = classify_island(area)
        if cat is None:
            continue
        kept.append({
            "bbox": cc["bbox"],
            "center": cc["center"],
            "area_px": area,
            "category": cat,
            "is_edge_truncated": bbox_touches_frame(cc["bbox"], water.shape, EDGE_MARGIN_PX),
        })
    return kept


# ---- Edge revalidation against z14 ----


def revalidate_edges_at_z14(candidates: list[dict],
                            area_id: str,
                            cell_id: str) -> tuple[int, int, int]:
    """For each is_edge_truncated island candidate, look it up in the wide
    z14 land mask. If the corresponding z14 land CC also doesn't touch the
    z14 frame (i.e., still bounded by water in the wider view), confirm it.
    Otherwise drop — it's a peninsula extending off-frame, not a real island.

    Mutates candidates in-place: sets confirmed_at_z14 on survivors and
    removes the dropped ones. Returns (n_edge_input, n_confirmed, n_dropped).
    No-op if the wide z14 styled image isn't on disk for this cell.
    """
    edge_indices = [i for i, c in enumerate(candidates) if c.get("is_edge_truncated")]
    if not edge_indices:
        return (0, 0, 0)

    z14_water = load_z14_water_mask(area_id, cell_id)
    if z14_water is None:
        print(f"  no z14 wide tile for {cell_id} — skipping edge revalidation")
        return (0, 0, 0)

    # Land CCs at z14, same no-smoothing pipeline as z16.
    z14_land = ~z14_water
    z14_land_ccs = connected_components(z14_land, min_pixels=MIN_ISLAND_AREA_PX // 16)
    z14_h, z14_w = z14_land.shape

    drop_indices: list[int] = []
    n_confirmed = 0

    for i in edge_indices:
        c = candidates[i]
        z14_cx, z14_cy = z16_to_z14(c["center"][0], c["center"][1])
        zx, zy = int(z14_cx), int(z14_cy)
        # Find the z14 land CC whose pixel set contains (zy, zx). If none
        # (e.g. the land got eroded away by smoothing at z14), drop.
        host = None
        for cc in z14_land_ccs:
            x0, y0, x1, y1 = cc["bbox"]
            if not (x0 <= zx < x1 and y0 <= zy < y1):
                continue
            if (zy, zx) in cc["pixels"]:
                host = cc
                break
        if host is None:
            drop_indices.append(i)
            continue
        if bbox_touches_frame(host["bbox"], (z14_h, z14_w), EDGE_MARGIN_PX):
            # Off-frame land continues at z14 -> peninsula, not island
            drop_indices.append(i)
        else:
            c["confirmed_at_z14"] = True
            n_confirmed += 1

    for i in sorted(drop_indices, reverse=True):
        candidates.pop(i)
    return (len(edge_indices), n_confirmed, len(drop_indices))


# ---- Rendering ----

CATEGORY_COLOR = {
    "ISLAND_SMALL":  (150, 230, 100, 255),   # light green
    "ISLAND_MEDIUM": ( 50, 180,  50, 255),   # medium green
    "ISLAND_LARGE":  ( 30, 110,  30, 255),   # dark green
}
LAND_TINT = (180, 140, 80, 50)               # soft tan tint to show the land mask


def render_overlay(base_image_path: Path,
                   candidates: list[dict],
                   land_mask: np.ndarray,
                   output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # Tint the land mask so reviewers can SEE which pixels the algorithm
    # treated as land when looking for islands.
    h, w = land_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[land_mask] = LAND_TINT
    land_layer = Image.fromarray(rgba, mode="RGBA")
    layer = Image.alpha_composite(layer, land_layer)
    draw = ImageDraw.Draw(layer, "RGBA")

    # 8x8 A1-H8 grid (same style as drains)
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
        x0 = max(0, bbox[0] - 2)
        y0 = max(0, bbox[1] - 2)
        x1 = min(image_size[0], bbox[2] + 2)
        y1 = min(image_size[1], bbox[3] + 2)
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
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(255, 255, 255, 255), width=2)
        edge_tag = "[E] " if is_edge else ""
        text = f"i{i} {edge_tag}{cand['category']} @{cell}  area={cand['area_px']}px"
        tx = max(2, min(image_size[0] - 360, int(cx) + 10))
        ty = max(2, min(image_size[1] - 22, int(cy) - 18))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=color, font=label_font)

    # Legend (top-right)
    legend_rows = [
        ("ISLAND_SMALL",  CATEGORY_COLOR["ISLAND_SMALL"],  f"area 100-{SMALL_MAX:,}px (~10-45 m diameter)"),
        ("ISLAND_MEDIUM", CATEGORY_COLOR["ISLAND_MEDIUM"], f"area {SMALL_MAX:,}-{MEDIUM_MAX:,}px (~45-140 m)"),
        ("ISLAND_LARGE",  CATEGORY_COLOR["ISLAND_LARGE"],  f"area {MEDIUM_MAX:,}-{MAX_ISLAND_AREA_PX:,}px (~140-450 m)"),
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
              "Islands  (kept categories only)",
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

    print(f"detecting islands (area {MIN_ISLAND_AREA_PX}-{MAX_ISLAND_AREA_PX:,} px)...")
    candidates = detect_islands(water)
    edge_count = sum(1 for c in candidates if c.get("is_edge_truncated"))
    print(f"  {len(candidates)} initial kept; {edge_count} edge-truncated")

    if edge_count:
        print(f"  revalidating edge candidates against wide z14 view...")
        n_in, n_ok, n_drop = revalidate_edges_at_z14(candidates, area_id, cell_id)
        print(f"  z14 revalidation: {n_in} edge candidates checked, "
              f"{n_ok} confirmed, {n_drop} dropped (peninsula off-frame)")

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"  {len(candidates)} final kept: "
          f"SMALL={counts.get('ISLAND_SMALL', 0)}  "
          f"MEDIUM={counts.get('ISLAND_MEDIUM', 0)}  "
          f"LARGE={counts.get('ISLAND_LARGE', 0)}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = storage.cell_artifact_path(area_id, cell_id, "islands", ts, "png")
    json_path = storage.cell_artifact_path(area_id, cell_id, "islands", ts, "json")

    land_mask = ~water
    render_overlay(z16_path, candidates, land_mask, overlay_path)

    storage.atomic_write_json(json_path, {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "detector": "cv_detect_islands",
        "min_area_px": MIN_ISLAND_AREA_PX,
        "max_area_px": MAX_ISLAND_AREA_PX,
        "candidates": [
            {
                "id": f"i{i}",
                "category": c["category"],
                "is_edge_truncated": c.get("is_edge_truncated", False),
                "confirmed_at_z14": c.get("confirmed_at_z14", False),
                "pixel_bbox": list(c["bbox"]),
                "pixel_center": [round(c["center"][0], 1), round(c["center"][1], 1)],
                "area_px": c["area_px"],
            }
            for i, c in enumerate(candidates, start=1)
        ],
    })

    if candidates:
        print(f"\n  {'id':<5s} {'cat':<14s} {'cell':<4s} {'edge':>5s} {'area':>8s}")
        for i, c in enumerate(candidates, start=1):
            cell = grid_cell_for(c["center"][0], c["center"][1])
            if c.get("is_edge_truncated"):
                edge_flag = "E*" if c.get("confirmed_at_z14") else "E?"
            else:
                edge_flag = "  "
            print(f"  i{i:<4d} {c['category']:<14s} {cell:<4s} {edge_flag:>5s} "
                  f"{c['area_px']:>6d}px")
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
