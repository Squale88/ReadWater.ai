"""CV-based drain candidate detection from a binary water mask.

Strategy: morphological opening with a kernel sized to the maximum "throat
width" we want to consider a drain. Wide water bodies survive the opening;
narrow connections (drain throats) get erased. Subtracting the opened mask
from the original gives us the narrow connections — drain throat candidates.

For each candidate region (connected component), emit a bounding box and
center pixel. The downstream LLM validator (next stage) will receive this
list with the satellite image and confirm/reject each.

Pure numpy + PIL — no scipy/skimage/cv2 required. Iterative 4-connected
erosion/dilation; slow per pixel but fine for one cell at a time.

Usage:
  python scripts/cv_detect_drains.py --cell root-10-8
  python scripts/cv_detect_drains.py --cell root-2-9 --max-throat-px 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# Pure-numpy morphology helpers
# ------------------------------------------------------------------


def erode_4conn(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Iteratively erode a boolean mask by 1 pixel per iteration, 4-connected.

    A pixel survives erosion only if all 4 cardinal neighbors are also True.
    Approximates a diamond-shaped structuring element of radius `iterations`.
    """
    m = mask.copy()
    for _ in range(iterations):
        # A pixel is True after erosion only if it AND all 4 neighbors were True.
        # Pad with False so edges erode inward.
        padded = np.pad(m, 1, constant_values=False)
        m = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]   # north neighbor
            & padded[2:, 1:-1]    # south neighbor
            & padded[1:-1, :-2]   # west neighbor
            & padded[1:-1, 2:]    # east neighbor
        )
    return m


def dilate_4conn(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Iteratively dilate a boolean mask by 1 pixel per iteration, 4-connected."""
    m = mask.copy()
    for _ in range(iterations):
        padded = np.pad(m, 1, constant_values=False)
        m = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return m


def connected_components(mask: np.ndarray, min_pixels: int = 10) -> list[dict]:
    """Find 4-connected components of a boolean mask. Returns list of dicts:
        {bbox: (x0, y0, x1, y1), center: (cx, cy), area: int}
    Filters out components smaller than `min_pixels`.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            # BFS
            stack = [(y, x)]
            min_x, min_y, max_x, max_y = x, y, x, y
            sum_x = sum_y = 0
            count = 0
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
                if xx < min_x: min_x = xx
                if xx > max_x: max_x = xx
                if yy < min_y: min_y = yy
                if yy > max_y: max_y = yy
                for dy, dx in nbrs:
                    stack.append((yy + dy, xx + dx))
            if count >= min_pixels:
                components.append({
                    "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                    "center": (sum_x / count, sum_y / count),
                    "area": count,
                })
    return components


# ------------------------------------------------------------------
# Drain detection
# ------------------------------------------------------------------


def smooth_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Closing then opening with a small kernel to clean fuzzy mask boundaries.

    Closing (dilate-then-erode) fills small holes; opening (erode-then-dilate)
    removes small protrusions. Net: smooths the boundary without changing
    the overall shape much.
    """
    closed = erode_4conn(dilate_4conn(mask, radius), radius)
    opened = dilate_4conn(erode_4conn(closed, radius), radius)
    return opened


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _bbox_touches_edge(bbox: tuple[int, int, int, int],
                       image_size: tuple[int, int],
                       margin: int = 4) -> bool:
    w, h = image_size
    return (bbox[0] <= margin or bbox[1] <= margin
            or bbox[2] >= w - margin or bbox[3] >= h - margin)


def detect_drain_candidates(
    water_mask: np.ndarray,
    max_throat_px: int = 25,
    min_component_px: int = 80,
    smooth_radius: int = 4,
    min_elongation: float = 1.5,
    edge_margin: int = 6,
) -> list[dict]:
    """Detect drain throat candidates via morphological opening, with filtering.

    A drain throat is a narrow water connection that disappears when we open
    the water mask with a kernel sized larger than the throat width.

    Filtering pipeline:
      1. Pre-smooth the mask to remove boundary noise (closing + opening with
         small kernel).
      2. Compute narrow = water - opened(water, max_throat_px / 2).
      3. Connected-component scan; drop components with area < min_component_px.
      4. Reject components whose bbox touches the image edge (mostly artifacts
         where the cell frame cuts a channel).
      5. Require elongation (max_dim / min_dim >= min_elongation) — drains are
         throats, not blobs.

    Args:
      water_mask: HxW boolean array; True = water, False = land.
      max_throat_px: max width considered a throat. Default 25 = ~25m drains.
      min_component_px: drop candidates smaller than this. Default 80px.
      smooth_radius: kernel radius for boundary smoothing. Default 4.
      min_elongation: max_bbox_dim / min_bbox_dim required. Default 1.5.
      edge_margin: drop candidates whose bbox touches within this many pixels
        of the image edge. Default 6.

    Returns: list of candidate dicts (bbox, center, area, elongation).
    """
    smoothed = smooth_mask(water_mask, radius=smooth_radius)
    radius = max(1, max_throat_px // 2)
    eroded = erode_4conn(smoothed, radius)
    opened = dilate_4conn(eroded, radius)
    narrow = smoothed & ~opened

    components = connected_components(narrow, min_pixels=min_component_px)
    h, w = water_mask.shape
    filtered: list[dict] = []
    for c in components:
        bbox = c["bbox"]
        if _bbox_touches_edge(bbox, (w, h), margin=edge_margin):
            continue
        bw, bh = _bbox_dims(bbox)
        long_dim = max(bw, bh)
        short_dim = max(1, min(bw, bh))
        elongation = long_dim / short_dim
        if elongation < min_elongation:
            continue
        c["elongation"] = round(elongation, 2)
        filtered.append(c)
    return filtered


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------


def _grid_cell_for(px: float, py: float,
                   image_size: tuple[int, int] = (1280, 1280),
                   rows: int = 8, cols: int = 8) -> str:
    """Return the A1-H8 cell label for a pixel position."""
    w, h = image_size
    col = max(0, min(cols - 1, int(px / (w / cols))))
    row = max(0, min(rows - 1, int(py / (h / rows))))
    return f"{chr(ord('A') + row)}{col + 1}"


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_candidates_overlay(
    base_image_path: Path,
    candidates: list[dict],
    output_path: Path,
    image_size: tuple[int, int] = (1280, 1280),
) -> Path:
    """Draw 8x8 A1-H8 grid + candidate boxes/centers/labels on the base image.

    Each candidate gets:
      - red bbox + center dot
      - label "dN @ <cell>" (the grid cell containing the center)
    """
    base = Image.open(base_image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    # 8x8 A1-H8 grid (same style as ensure_grid_overlay)
    rows, cols = 8, 8
    gw, gh = image_size
    cell_w = gw / cols
    cell_h = gh / rows
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
            label = f"{chr(ord('A') + r)}{c + 1}"
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            bbox_t = grid_font.getbbox(label)
            tw = bbox_t[2] - bbox_t[0]
            th = bbox_t[3] - bbox_t[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, fill="black", font=grid_font)
            draw.text((tx, ty), label, fill="white", font=grid_font)

    # Candidate markings on top of the grid
    label_font = _load_font(20)
    for i, c in enumerate(candidates, start=1):
        bbox = c["bbox"]
        cx, cy = c["center"]
        cell = _grid_cell_for(cx, cy, image_size)
        x0 = max(0, bbox[0] - 4)
        y0 = max(0, bbox[1] - 4)
        x1 = min(base.size[0], bbox[2] + 4)
        y1 = min(base.size[1], bbox[3] + 4)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 60, 60, 255), width=3)
        r = 9
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(255, 60, 60, 255), outline=(255, 255, 255, 255), width=2)
        # Label with id + cell + area, with a black halo for legibility over imagery
        text = f"d{i} @ {cell}  ({c['area']}px)"
        tx = max(2, min(image_size[0] - 220, int(cx) + 12))
        ty = max(2, min(image_size[1] - 24, int(cy) - 22))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=label_font)

    Image.alpha_composite(base, layer).convert("RGB").save(output_path)
    return output_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, help="cell_id like root-10-8")
    parser.add_argument("--max-throat-px", type=int, default=25,
                        help="Max throat width in px to flag (~m at z16). Default 25.")
    parser.add_argument("--min-component-px", type=int, default=80,
                        help="Drop candidates smaller than this (pixels). Default 80.")
    parser.add_argument("--smooth-radius", type=int, default=4,
                        help="Kernel radius for boundary smoothing. Default 4.")
    parser.add_argument("--min-elongation", type=float, default=1.5,
                        help="Required max_dim/min_dim ratio. Default 1.5.")
    parser.add_argument("--edge-margin", type=int, default=6,
                        help="Drop candidates within N px of image edge. Default 6.")
    args = parser.parse_args()

    cell_id = args.cell
    parent_num, child_num = cell_id.removeprefix("root-").split("-")
    stem = f"z0_{parent_num}_{child_num}"

    water_mask_path = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2_google_water"
                       / f"{cell_id}_water_mask.png")
    z16_path = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
                / f"{stem}.png")
    out_dir = (REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
               / "structures" / cell_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not water_mask_path.exists():
        print(f"missing water mask: {water_mask_path}", file=sys.stderr)
        return 2
    if not z16_path.exists():
        print(f"missing z16 image: {z16_path}", file=sys.stderr)
        return 2

    print(f"loading water mask: {water_mask_path.name}")
    mask_img = Image.open(water_mask_path).convert("L")
    if mask_img.size != (1280, 1280):
        print(f"resizing mask {mask_img.size} -> (1280, 1280)")
        mask_img = mask_img.resize((1280, 1280), Image.NEAREST)
    water = np.array(mask_img) > 127

    print(f"detecting candidates (max_throat={args.max_throat_px}px, "
          f"smooth={args.smooth_radius}, min_area={args.min_component_px}, "
          f"min_elong={args.min_elongation})...")
    candidates = detect_drain_candidates(
        water,
        max_throat_px=args.max_throat_px,
        min_component_px=args.min_component_px,
        smooth_radius=args.smooth_radius,
        min_elongation=args.min_elongation,
        edge_margin=args.edge_margin,
    )
    print(f"detected {len(candidates)} drain throat candidates")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = out_dir / f"cv_drain_candidates_{ts}.png"
    json_path = out_dir / f"cv_drain_candidates_{ts}.json"
    render_candidates_overlay(z16_path, candidates, overlay_path)
    json_path.write_text(json.dumps({
        "cell_id": cell_id,
        "max_throat_px": args.max_throat_px,
        "min_component_px": args.min_component_px,
        "candidates": [
            {
                "candidate_id": f"d{i}",
                "pixel_bbox": list(c["bbox"]),
                "pixel_center": [round(c["center"][0], 1), round(c["center"][1], 1)],
                "area_px": c["area"],
            }
            for i, c in enumerate(candidates, start=1)
        ],
    }, indent=2), encoding="utf-8")

    print()
    print(f"=== {cell_id} drain candidates ===")
    print(f"  {'id':<4s} {'cell':<4s} {'center':<14s} {'bbox':<24s} {'area':>6s} {'elong':>5s}")
    for i, c in enumerate(candidates, start=1):
        bbox = c["bbox"]
        cx, cy = c["center"]
        cell = _grid_cell_for(cx, cy)
        elong = c.get("elongation", 0.0)
        print(f"  d{i:<3d} {cell:<4s} ({cx:>4.0f},{cy:>4.0f})  "
              f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})  "
              f"{c['area']:>5d}  {elong:>5.2f}")
    print()
    print(f"overlay: {overlay_path.relative_to(REPO_ROOT)}")
    print(f"json:    {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
