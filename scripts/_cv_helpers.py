"""Shared building blocks for CV-based feature detectors.

All scripts that operate on the per-cell water mask share the same
morphology, connected-component, adjacency, and z16<->z14 plumbing. Pulling
those into one module keeps each detector script focused on the category
logic specific to its feature type.

Used by:
  - cv_detect_drains.py    (DRAIN / CREEK_MOUTH / LARGE_POCKET / SHOAL)
  - cv_detect_islands.py   (ISLAND_SMALL / _MEDIUM / _LARGE)
  - cv_detect_points.py    (POINT)
  - cv_detect_cuts.py      (CUT)
  - cv_detect_all.py       (orchestrator)

No detector logic lives here — only generic primitives. Each detector
imports what it needs and adds its own classify() / render_overlay() / CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- Shared constants ----

EDGE_MARGIN_PX = 6         # bbox within this many pixels of the frame -> is_edge_truncated
SMOOTH_RADIUS = 4          # boundary smoothing kernel applied to every mask
GOOGLE_WATER_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_google_water"


# ---- Pure-numpy 4-connected morphology ----


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


def smooth_mask(mask: np.ndarray, radius: int = SMOOTH_RADIUS) -> np.ndarray:
    """Closing then opening at small radius to clean noisy mask boundaries."""
    closed = erode_4conn(dilate_4conn(mask, radius), radius)
    return dilate_4conn(erode_4conn(closed, radius), radius)


# ---- Connected components + adjacency ----


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


def find_adjacent(seed_pixels: set[tuple[int, int]],
                  components: list[dict],
                  shape: tuple[int, int]) -> list[dict]:
    """Return CCs that touch (within 1 px of) the seed region."""
    h, w = shape
    cand_mask = np.zeros((h, w), dtype=bool)
    for (y, x) in seed_pixels:
        cand_mask[y, x] = True
    dilated = dilate_4conn(cand_mask, 1)
    adjacent = []
    for comp in components:
        x0, y0, x1, y1 = comp["bbox"]
        comp_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for (y, x) in comp["pixels"]:
            comp_mask[y - y0, x - x0] = True
        if (comp_mask & dilated[y0:y1, x0:x1]).any():
            adjacent.append(comp)
    return adjacent


# ---- Density measurement (used by drains for SHOAL test, also useful elsewhere) ----


def water_density_around(water: np.ndarray,
                         center: tuple[float, float],
                         radius: int) -> float:
    """Fraction of pixels in a square neighborhood that are water."""
    h, w = water.shape
    cx, cy = int(center[0]), int(center[1])
    x0 = max(0, cx - radius); x1 = min(w, cx + radius)
    y0 = max(0, cy - radius); y1 = min(h, cy + radius)
    region = water[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


# ---- Edge / cell-frame detection ----


def bbox_touches_frame(bbox: tuple[int, int, int, int],
                       shape: tuple[int, int],
                       margin: int = EDGE_MARGIN_PX) -> bool:
    """True if the bbox sits within `margin` pixels of any image edge."""
    h, w = shape
    return (
        bbox[0] <= margin
        or bbox[1] <= margin
        or bbox[2] >= w - margin
        or bbox[3] >= h - margin
    )


# ---- Z16 <-> Z14 coordinate mapping ----


def z16_to_z14(z16_x: float, z16_y: float) -> tuple[float, float]:
    """Map z16 pixel coords to z14 pixel coords.

    Both Static-Maps tiles are 1280x1280 centered on the same lat/lon. z14
    covers 4x the linear extent (16x area), so the z16 cell footprint occupies
    the central 320x320 region of the z14 image (pixels 480..800 in each axis).
    """
    return (480.0 + z16_x / 4.0, 480.0 + z16_y / 4.0)


# ---- Z14 mask loading ----


def water_mask_from_styled_png(styled_png: Path) -> np.ndarray:
    """Threshold a styled water tile (water=blue, land=white) into a binary
    boolean mask. Mirrors scripts/google_water_mask.py:water_mask_from_styled.
    """
    img = Image.open(styled_png).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (b > 128) & (r < 96) & (g < 96)


def load_z14_water_mask(cell_id: str) -> np.ndarray | None:
    """Load and threshold the wide z14 styled tile for a cell.

    Returns a 1280x1280 boolean array (True = water) or None if the file
    isn't on disk. The styled tile lives next to the z16 mask under
    ``data/areas/rookery_bay_v2_google_water/<cell>_wide_z14_styled.png``
    and was produced by ``scripts/google_water_mask.py --wide``.
    """
    z14_styled = GOOGLE_WATER_DIR / f"{cell_id}_wide_z14_styled.png"
    if not z14_styled.exists():
        return None
    mask = water_mask_from_styled_png(z14_styled)
    if mask.shape != (1280, 1280):
        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        img = img.resize((1280, 1280), Image.NEAREST)
        mask = np.array(img) > 127
    return mask


# ---- Grid-cell labels (8x8 A1-H8) ----


def grid_cell_for(px: float, py: float,
                  image_size: tuple[int, int] = (1280, 1280),
                  rows: int = 8, cols: int = 8) -> str:
    """Return the A1..H8 grid label containing the given pixel."""
    w, h = image_size
    col = max(0, min(cols - 1, int(px / (w / cols))))
    row = max(0, min(rows - 1, int(py / (h / rows))))
    return f"{chr(ord('A') + row)}{col + 1}"


# ---- Font loading ----


def load_font(size: int):
    """Best-effort truetype font lookup with a final fallback to PIL default."""
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()
