"""CV-based detection of fishable water features (phased build).

PHASE 1 — medium drains (throats 25-50 px ≈ 25-50 m at z16).

We start with medium because in our test set (root-10-8 / 2-9 / 11-1 / 11-5)
the dominant visible drain throats are in the 25-60 px band; small drains
(< 25 px) are rare in the Google water mask because canopy hides their
back-water. We will layer Small as a second pass after Medium is dialed in.

A medium drain throat is:
  * A narrow water region (≤ ~50 px wide) that, when removed from the water
    mask, separates the surrounding water into two distinct bodies.
  * Exactly one of those bodies is OPEN (touches an image edge — assumed
    tidally connected to the broader system).
  * The other body is ENCLOSED (does not touch any image edge) and has
    enclosed area ≥ 20,000 px (the Medium drain minimum back-area volume).

Input: the Google z16 water mask
  data/areas/rookery_bay_v2_google_water/<cell>_water_mask.png

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

# ---- Phase 1 (medium drain) parameters ----

PHASE1_RADIUS = 25             # captures throats up to ~50 px wide
PHASE1_MIN_THROAT_PX = 25      # narrower belongs to Small (later pass)
PHASE1_MAX_THROAT_PX = 60      # wider belongs to Large (later pass)
PHASE1_MIN_NARROW_AREA = 200   # min pixel area of a narrow CC to consider
PHASE1_MIN_ELONGATION = 1.5    # bbox long/short ratio — throats not blobs
PHASE1_MIN_ENCLOSED_AREA = 20_000   # min back-area for a Medium drain (per spec)
PHASE1_MAX_ENCLOSED_AREA = 250_000  # > this defers to Large/XL (later pass)
EDGE_MARGIN_PX = 6             # near-edge bboxes count as edge-touching
SMOOTH_RADIUS = 4              # boundary smoothing kernel for the water mask

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


def touches_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    return (
        bbox[0] <= EDGE_MARGIN_PX
        or bbox[1] <= EDGE_MARGIN_PX
        or bbox[2] >= w - EDGE_MARGIN_PX
        or bbox[3] >= h - EDGE_MARGIN_PX
    )


# ---- Phase 1 detection ----


def detect_medium_drains(water: np.ndarray) -> list[dict]:
    """Phase 1: throats 25-50 px wide that connect open water to an enclosed back-area
    of at least PHASE1_MIN_ENCLOSED_AREA px."""
    h, w = water.shape
    smoothed = smooth_mask(water, radius=SMOOTH_RADIUS)
    wide = open_mask(smoothed, PHASE1_RADIUS)
    narrow = smoothed & ~wide

    narrow_ccs = connected_components(narrow, min_pixels=PHASE1_MIN_NARROW_AREA)
    wide_ccs = connected_components(wide, min_pixels=200)

    drains: list[dict] = []
    rejected: list[dict] = []   # for debug — not rendered

    for nc in narrow_ccs:
        bbox = nc["bbox"]
        if touches_edge(bbox, water.shape):
            rejected.append({"reason": "narrow CC at frame edge", "bbox": bbox})
            continue
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        long_dim = max(bw, bh)
        short_dim = max(1, min(bw, bh))
        elong = long_dim / short_dim
        throat_width = short_dim
        if elong < PHASE1_MIN_ELONGATION:
            rejected.append({"reason": f"elong {elong:.2f} < {PHASE1_MIN_ELONGATION}",
                             "bbox": bbox})
            continue
        if throat_width < PHASE1_MIN_THROAT_PX or throat_width > PHASE1_MAX_THROAT_PX:
            rejected.append({"reason": f"throat {throat_width}px outside medium range",
                             "bbox": bbox})
            continue

        adj = find_adjacent(nc["pixels"], wide_ccs, water.shape)
        if len(adj) < 2:
            rejected.append({"reason": f"only {len(adj)} adjacent wide(s)",
                             "bbox": bbox})
            continue

        # Sort by area descending; pick the two largest as the canonical sides.
        adj.sort(key=lambda c: c["area"], reverse=True)
        a, b = adj[0], adj[1]
        a_open = touches_edge(a["bbox"], water.shape)
        b_open = touches_edge(b["bbox"], water.shape)

        # Phase 1 = exactly one open + one enclosed
        if a_open and b_open:
            rejected.append({"reason": "both sides touch edge (cut/pass, not drain)",
                             "bbox": bbox, "a_area": a["area"], "b_area": b["area"]})
            continue
        if not a_open and not b_open:
            rejected.append({"reason": "both sides enclosed (trapped)",
                             "bbox": bbox, "a_area": a["area"], "b_area": b["area"]})
            continue

        enclosed = a if not a_open else b
        open_side = b if not a_open else a
        ea = enclosed["area"]
        if ea < PHASE1_MIN_ENCLOSED_AREA:
            rejected.append({"reason": f"enclosed {ea}px < {PHASE1_MIN_ENCLOSED_AREA} (volume too small)",
                             "bbox": bbox, "throat_px": throat_width})
            continue
        if ea > PHASE1_MAX_ENCLOSED_AREA:
            rejected.append({"reason": f"enclosed {ea}px > {PHASE1_MAX_ENCLOSED_AREA} (defer to higher tier)",
                             "bbox": bbox, "throat_px": throat_width})
            continue

        drains.append({
            "bbox": bbox,
            "center": nc["center"],
            "area_px": nc["area"],
            "throat_width_px": throat_width,
            "elongation": round(elong, 2),
            "enclosed_area_px": ea,
            "open_area_px": open_side["area"],
            "classification": "DRAIN",
            "tier": "Medium",
            "reason": (f"throat {throat_width}px; enclosed back-area {ea}px "
                       f"(>= {PHASE1_MIN_ENCLOSED_AREA}); open side touches frame"),
        })

    return drains, rejected


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


MEDIUM_DRAIN_COLOR = (220, 30, 30, 255)   # red
REJECT_COLORS = {
    # Color-code rejection reasons so the overlay tells us which filter is firing.
    "edge":           (255, 165,   0, 220),  # orange — frame edge
    "elong":          (255, 230,   0, 220),  # yellow — low elongation
    "throat_range":   (200, 100, 255, 220),  # purple — throat width outside 5-25
    "adjacents":      (  0, 200, 255, 220),  # cyan  — only 0/1 adjacent wides
    "enclosed_small": (255,  60, 200, 220),  # pink  — enclosed back-area < 2k
    "enclosed_big":   (130, 200,  80, 220),  # green — enclosed > 50k (defer)
    "both_open":      (180, 180, 255, 220),  # light blue — both sides at edge
    "both_enclosed":  ( 90,  90,  90, 220),  # gray  — trapped pool
}


def _bucket_for_reason(reason: str) -> str:
    if reason.startswith("narrow CC at frame edge"):       return "edge"
    if reason.startswith("elong "):                        return "elong"
    if reason.startswith("throat "):                       return "throat_range"
    if reason.startswith("only "):                         return "adjacents"
    if reason.startswith("enclosed ") and "< " in reason:  return "enclosed_small"
    if reason.startswith("enclosed ") and "> " in reason:  return "enclosed_big"
    if reason.startswith("both sides touch"):              return "both_open"
    if reason.startswith("both sides enclosed"):           return "both_enclosed"
    return "edge"


def render_overlay(base_image_path: Path, drains: list[dict],
                   rejected: list[dict], output_path: Path,
                   image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

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

    label_font = _load_font(18)
    small_font = _load_font(11)
    # Draw rejected candidates first (thinner outline, no fill) so accepted drains
    # render on top.
    for r in rejected:
        bbox = r.get("bbox")
        if not bbox:
            continue
        bucket = _bucket_for_reason(r.get("reason", ""))
        color = REJECT_COLORS.get(bucket, (180, 180, 180, 200))
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1],
                       outline=color, width=1)
        # Tiny letter tag in top-left of the rejected bbox to identify the bucket.
        tag = bucket[:1].upper() if bucket else "?"
        draw.text((bbox[0] + 2, bbox[1] + 1), tag, fill=color, font=small_font)

    for i, d in enumerate(drains, start=1):
        bbox = d["bbox"]
        cx, cy = d["center"]
        cell = _grid_cell_for(cx, cy, image_size)
        x0 = max(0, bbox[0] - 4)
        y0 = max(0, bbox[1] - 4)
        x1 = min(image_size[0], bbox[2] + 4)
        y1 = min(image_size[1], bbox[3] + 4)
        draw.rectangle([x0, y0, x1, y1], outline=MEDIUM_DRAIN_COLOR, width=3)
        r = 8
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=MEDIUM_DRAIN_COLOR,
                     outline=(255, 255, 255, 255), width=2)
        text = (f"d{i} MEDIUM @ {cell}  "
                f"throat={d['throat_width_px']}px  back={d['enclosed_area_px']}px")
        tx = max(2, min(image_size[0] - 360, int(cx) + 12))
        ty = max(2, min(image_size[1] - 24, int(cy) - 22))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=MEDIUM_DRAIN_COLOR, font=label_font)

    # Legend in top-right (accepted + reject buckets)
    legend = [
        (MEDIUM_DRAIN_COLOR, "MEDIUM DRAIN (25-50px / back 20k-250k)"),
        (REJECT_COLORS["edge"],           "E  reject: at frame edge"),
        (REJECT_COLORS["elong"],          "L  reject: low elongation"),
        (REJECT_COLORS["throat_range"],   "T  reject: throat outside 5-25"),
        (REJECT_COLORS["adjacents"],      "A  reject: < 2 adjacent wides"),
        (REJECT_COLORS["enclosed_small"], f"S  reject: back-area < {PHASE1_MIN_ENCLOSED_AREA}"),
        (REJECT_COLORS["enclosed_big"],   f"B  reject: back-area > {PHASE1_MAX_ENCLOSED_AREA} (defer)"),
        (REJECT_COLORS["both_open"],      "O  reject: both sides at edge"),
        (REJECT_COLORS["both_enclosed"],  "X  reject: both sides enclosed"),
    ]
    line_h = 20
    box_w = 360
    box_h = line_h * len(legend) + 12
    bx0 = image_size[0] - box_w - 8
    by0 = 8
    draw.rectangle([bx0, by0, bx0 + box_w, by0 + box_h],
                   fill=(0, 0, 0, 210), outline=(255, 255, 255, 255), width=2)
    ty = by0 + 6
    for color, text in legend:
        sx = bx0 + 8
        draw.rectangle([sx, ty + 2, sx + 16, ty + 16], fill=color)
        draw.text((sx + 24, ty), text, fill=(255, 255, 255, 255), font=small_font)
        ty += line_h

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

    print(f"detecting medium drains (R={PHASE1_RADIUS}, throat 25-50px, "
          f"back-area >= {PHASE1_MIN_ENCLOSED_AREA})...")
    drains, rejected = detect_medium_drains(water)
    print(f"  {len(drains)} medium drain(s) found, {len(rejected)} narrow CCs rejected")
    # Summarize rejection reasons (group by leading prefix to fold "X px < Y" together)
    from collections import Counter
    reason_buckets = Counter()
    for r in rejected:
        reason = r["reason"]
        # Bucket by the first few words (skip the numeric tail)
        if reason.startswith("only "):
            key = "only N adjacent wide(s)"
        elif reason.startswith("elong "):
            key = "low elongation"
        elif reason.startswith("throat "):
            key = "throat outside medium range (25-50px)"
        elif reason.startswith("enclosed ") and "< " in reason:
            key = f"enclosed area < {PHASE1_MIN_ENCLOSED_AREA}px (too small)"
        elif reason.startswith("enclosed ") and "> " in reason:
            key = f"enclosed area > {PHASE1_MAX_ENCLOSED_AREA}px (defer to Large/XL)"
        else:
            key = reason
        reason_buckets[key] += 1
    print(f"  rejection reasons:")
    for k, v in sorted(reason_buckets.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>3d}  {k}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = out_dir / f"cv_phase1_medium_drains_{ts}.png"
    json_path = out_dir / f"cv_phase1_medium_drains_{ts}.json"

    render_overlay(z16_path, drains, rejected, overlay_path)
    json_path.write_text(json.dumps({
        "cell_id": cell_id,
        "phase": 1,
        "phase_params": {
            "scale_radius_px": PHASE1_RADIUS,
            "min_throat_px": PHASE1_MIN_THROAT_PX,
            "max_throat_px": PHASE1_MAX_THROAT_PX,
            "min_enclosed_area_px": PHASE1_MIN_ENCLOSED_AREA,
            "max_enclosed_area_px": PHASE1_MAX_ENCLOSED_AREA,
        },
        "drains": [
            {
                "id": f"d{i}",
                "classification": d["classification"],
                "tier": d["tier"],
                "throat_width_px": d["throat_width_px"],
                "pixel_bbox": list(d["bbox"]),
                "pixel_center": [round(d["center"][0], 1), round(d["center"][1], 1)],
                "enclosed_area_px": d["enclosed_area_px"],
                "open_area_px": d["open_area_px"],
                "elongation": d["elongation"],
                "reason": d["reason"],
            }
            for i, d in enumerate(drains, start=1)
        ],
        "rejected_count": len(rejected),
    }, indent=2), encoding="utf-8")

    if drains:
        print(f"\n  {'id':<4s} {'cell':<4s} {'throat':>6s} {'back':>7s}  reason")
        for i, d in enumerate(drains, start=1):
            cell = _grid_cell_for(d["center"][0], d["center"][1])
            print(f"  d{i:<3d} {cell:<4s} "
                  f"{d['throat_width_px']:>4d}px {d['enclosed_area_px']:>6d}px  "
                  f"{d['reason']}")
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
