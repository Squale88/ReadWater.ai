"""POC: classical CV segmentation of the zoom-18 mosaic.

Classifies every pixel into one of {water, mangrove, sand/shoal, mud} by
HSV thresholding, runs connected components, filters tiny regions, and
renders a 'menu' image with each surviving region outlined and numbered.

No new runtime deps (Pillow + numpy only — numpy is already transitive
via Pillow and used by scikit-image-style workflows).

Run:
  python scripts/poc_cv_segmentation.py
  python scripts/poc_cv_segmentation.py --min-area 1500  # tune if too many/few regions
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

A2_MOSAIC = (
    REPO_ROOT
    / "data/areas/rookery_bay_v2_structure_test/images/cells/root-10-8/structures/a2/mosaic.png"
)
OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_grid_poc"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- Segmentation ----------

# Class colors for the menu overlay (RGB).
CLASS_COLORS = {
    "water": (50, 140, 230),
    "mangrove": (30, 130, 60),
    "sand": (240, 210, 120),
    "mud": (160, 120, 70),
}


def classify_pixels(hsv: np.ndarray) -> np.ndarray:
    """Return a per-pixel int8 class map.

    0 = unclassified, 1 = water, 2 = mangrove, 3 = sand, 4 = mud.

    Thresholds tuned empirically for coastal satellite imagery (Rookery Bay
    family). Pillow HSV byte space (0-255 per channel).

    Hue wheel in Pillow byte space:
        0    = red
       ~21   = orange
       ~42   = yellow
       ~85   = green
       ~127  = cyan
       ~170  = blue
       ~212  = magenta
    """
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    # Water: broad blue-cyan-green hues, full value range. Rookery Bay water
    # ranges from deep blue-green to very light greenish in the shoal areas.
    water = (
        (h >= 90) & (h <= 190)
        & (v >= 40)
        & (s >= 15)
    )

    # Sand / shoal: low saturation + high brightness. Very distinctive.
    sand = (s < 70) & (v > 140)

    # Mangrove / dense vegetation: broad green hue, any brightness, anything
    # not already water or sand. We include "dark green through olive brown"
    # because coastal mangroves render darker than pure grass-green.
    mangrove = (
        (h >= 25) & (h <= 110)
        & (v < 130)
        & (s >= 10)
    )

    # Mud / tidal flat: yellow-orange-brown hues, mid brightness.
    mud = (
        (h >= 15) & (h <= 45)
        & (v >= 60) & (v <= 180)
        & (s >= 25)
    )

    out = np.zeros(h.shape, dtype=np.int8)
    # Precedence: water (most distinctive) > sand > mangrove > mud.
    out[mud] = 4
    out[mangrove] = 2
    out[sand] = 3
    out[water] = 1
    return out


def morphological_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Cheap 4-connectivity erode-then-dilate to kill salt-and-pepper noise.

    Pure numpy via np.roll; fast enough on 800x800 images.
    """
    m = mask
    for _ in range(iters):
        # Erode: pixel stays True iff all 4-neighbors are True
        up = np.roll(m, -1, axis=0); up[-1, :] = False
        dn = np.roll(m, 1, axis=0); dn[0, :] = False
        lt = np.roll(m, -1, axis=1); lt[:, -1] = False
        rt = np.roll(m, 1, axis=1); rt[:, 0] = False
        m = m & up & dn & lt & rt
    for _ in range(iters):
        # Dilate: pixel becomes True if any 4-neighbor is True
        up = np.roll(m, -1, axis=0); up[-1, :] = False
        dn = np.roll(m, 1, axis=0); dn[0, :] = False
        lt = np.roll(m, -1, axis=1); lt[:, -1] = False
        rt = np.roll(m, 1, axis=1); rt[:, 0] = False
        m = m | up | dn | lt | rt
    return m


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connectivity labeling. Pure numpy + collections.deque BFS.

    Returns (labels, n_labels). 0 = background, 1..n = component IDs.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 0
    # A view as list-of-bools for speed in the inner loop.
    m = mask

    for y0 in range(H):
        row = m[y0]
        lrow = labels[y0]
        for x0 in range(W):
            if row[x0] and lrow[x0] == 0:
                next_label += 1
                q = deque()
                q.append((y0, x0))
                labels[y0, x0] = next_label
                while q:
                    y, x = q.popleft()
                    if y > 0 and m[y - 1, x] and labels[y - 1, x] == 0:
                        labels[y - 1, x] = next_label
                        q.append((y - 1, x))
                    if y + 1 < H and m[y + 1, x] and labels[y + 1, x] == 0:
                        labels[y + 1, x] = next_label
                        q.append((y + 1, x))
                    if x > 0 and m[y, x - 1] and labels[y, x - 1] == 0:
                        labels[y, x - 1] = next_label
                        q.append((y, x - 1))
                    if x + 1 < W and m[y, x + 1] and labels[y, x + 1] == 0:
                        labels[y, x + 1] = next_label
                        q.append((y, x + 1))
    return labels, next_label


def region_summary(labels: np.ndarray, n: int) -> list[dict]:
    """Return {label, area, centroid, bbox} for each region 1..n."""
    # Vectorized: use np.where once per label is O(n_pixels * n_labels), slow.
    # Instead, iterate pixels once and accumulate.
    H, W = labels.shape
    counts = np.zeros(n + 1, dtype=np.int64)
    sum_x = np.zeros(n + 1, dtype=np.int64)
    sum_y = np.zeros(n + 1, dtype=np.int64)
    min_x = np.full(n + 1, W, dtype=np.int32)
    max_x = np.full(n + 1, -1, dtype=np.int32)
    min_y = np.full(n + 1, H, dtype=np.int32)
    max_y = np.full(n + 1, -1, dtype=np.int32)

    # Numpy-friendly accumulation using bincount and np.where scans.
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n + 1)

    # For centroid and bbox, do per-label via np.argwhere-style.
    regions = []
    for lbl in range(1, n + 1):
        if counts[lbl] == 0:
            continue
        ys, xs = np.where(labels == lbl)
        cx = int(xs.mean())
        cy = int(ys.mean())
        regions.append({
            "label": lbl,
            "area": int(counts[lbl]),
            "centroid": (cx, cy),
            "bbox": (int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)),
        })
    return regions


# ---------- Rendering ----------


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_menu(
    base_image: Image.Image,
    class_map: np.ndarray,
    all_labels: dict[str, tuple[np.ndarray, list[dict]]],
) -> Image.Image:
    """Overlay each kept region as a translucent fill + solid outline + ID.

    `all_labels` is keyed by class name → (labels_array, region_summaries).
    Each region gets a unique global ID (running counter across all classes).
    """
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font_big = _load_font(26)
    font_small = _load_font(14)

    global_id = 0
    id_index: list[dict] = []

    for class_name, (labels, regions) in all_labels.items():
        color = CLASS_COLORS[class_name]
        for r in regions:
            global_id += 1
            # Translucent fill covering the region
            mask = (labels == r["label"])
            ys, xs = np.where(mask)
            for x, y in zip(xs, ys):
                overlay.putpixel((int(x), int(y)), (*color, 80))

            # Outline: boundary pixels (pixels where neighbor is not in region)
            # Efficient: shift the mask and XOR
            up = np.zeros_like(mask)
            up[1:, :] = mask[:-1, :]
            down = np.zeros_like(mask)
            down[:-1, :] = mask[1:, :]
            left = np.zeros_like(mask)
            left[:, 1:] = mask[:, :-1]
            right = np.zeros_like(mask)
            right[:, :-1] = mask[:, 1:]
            boundary = mask & ~(up & down & left & right)
            bys, bxs = np.where(boundary)
            for x, y in zip(bxs, bys):
                overlay.putpixel((int(x), int(y)), (*color, 255))

            # ID label at centroid
            cx, cy = r["centroid"]
            txt = str(global_id)
            bbox = font_big.getbbox(txt)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad = 4
            box_coords = (
                cx - tw // 2 - pad, cy - th // 2 - pad,
                cx + tw // 2 + pad, cy + th // 2 + pad,
            )
            draw.rectangle(box_coords, fill=(0, 0, 0, 230), outline=color, width=2)
            draw.text((cx - tw // 2, cy - th // 2 - 2), txt,
                      fill="white", font=font_big)

            id_index.append({
                "id": global_id,
                "class": class_name,
                "area": r["area"],
                "centroid": r["centroid"],
                "bbox": r["bbox"],
            })

    # Legend bottom-left
    legend_items = [("water", CLASS_COLORS["water"]),
                    ("mangrove", CLASS_COLORS["mangrove"]),
                    ("sand/shoal", CLASS_COLORS["sand"]),
                    ("mud/flat", CLASS_COLORS["mud"])]
    lx, ly = 16, base.size[1] - 16 - 26 * len(legend_items) - 14
    lw = 200
    lh = 26 * len(legend_items) + 14
    draw.rectangle([lx, ly, lx + lw, ly + lh], fill=(0, 0, 0, 200))
    for i, (name, col) in enumerate(legend_items):
        y = ly + 7 + i * 26
        draw.rectangle([lx + 8, y + 3, lx + 28, y + 20], fill=col)
        draw.text((lx + 40, y + 2), name, fill="white", font=font_small)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return final, id_index


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area", type=int, default=1500,
                        help="drop regions smaller than this (px)")
    parser.add_argument("--downscale", type=int, default=1,
                        help="downscale factor for speed (1 = full res)")
    args = parser.parse_args()

    if not A2_MOSAIC.exists():
        raise SystemExit(f"a2 mosaic not found at {A2_MOSAIC}")

    print(f"=== CV segmentation POC: a2 mosaic ===")
    img = Image.open(A2_MOSAIC).convert("RGB")
    print(f"loaded mosaic: {img.size}")
    if args.downscale > 1:
        img = img.resize(
            (img.size[0] // args.downscale, img.size[1] // args.downscale),
            Image.LANCZOS,
        )
        print(f"downscaled to: {img.size}")

    hsv_img = img.convert("HSV")
    hsv = np.array(hsv_img)
    print("classifying pixels...")
    class_map = classify_pixels(hsv)
    total = class_map.size
    print(f"  water:    {(class_map == 1).sum() / total:.1%}")
    print(f"  mangrove: {(class_map == 2).sum() / total:.1%}")
    print(f"  sand:     {(class_map == 3).sum() / total:.1%}")
    print(f"  mud:      {(class_map == 4).sum() / total:.1%}")
    print(f"  unclassified: {(class_map == 0).sum() / total:.1%}")

    all_labels: dict[str, tuple[np.ndarray, list[dict]]] = {}
    for class_id, class_name in [(1, "water"), (2, "mangrove"), (3, "sand"), (4, "mud")]:
        mask = (class_map == class_id)
        if not mask.any():
            continue
        print(f"labeling {class_name} ...")
        # Morphological open to kill salt-and-pepper noise from color thresholding
        mask = morphological_open(mask, iters=2)
        labels, n = connected_components(mask)
        print(f"  raw regions (after opening): {n}")
        regions = region_summary(labels, n)
        kept = [r for r in regions if r["area"] >= args.min_area]
        print(f"  kept (area >= {args.min_area}): {len(kept)}")
        all_labels[class_name] = (labels, kept)

    print("rendering menu image...")
    menu_img, id_index = render_menu(img, class_map, all_labels)
    menu_path = OUT_ROOT / "a2_cv_menu.png"
    menu_img.save(menu_path)
    print(f"menu image: {menu_path}")

    print()
    print("=== region index ===")
    print(f"{'id':>3}  {'class':<10}  {'area':>8}  centroid")
    for r in id_index:
        print(f"{r['id']:>3}  {r['class']:<10}  {r['area']:>8}  {r['centroid']}")


if __name__ == "__main__":
    main()
