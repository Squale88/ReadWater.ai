"""Grid overlay utilities — draw labeled grids, parse cell labels, convert
cell sets to pixel geometry.

Labels follow a spreadsheet convention:
  - Rows: A..Z (top to bottom), then AA..AZ, BA..BZ, ... for grids > 26
  - Cols: 1..N (left to right)
  - A1 is always top-left.

Pure geometry + Pillow drawing helpers. No I/O beyond writing the rendered
grid PNG; no external dependencies aside from Pillow.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# --- Label math ---


def row_label(idx: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB, ... (Excel-style)."""
    if idx < 0:
        raise ValueError(f"negative row index {idx}")
    if idx < 26:
        return string.ascii_uppercase[idx]
    first = (idx // 26) - 1
    second = idx % 26
    return string.ascii_uppercase[first] + string.ascii_uppercase[second]


_CELL_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")


def parse_cell(label: str) -> tuple[int, int] | None:
    """'A1' -> (row=0, col=0); 'H8' -> (7, 7); 'AB12' -> (27, 11). None on junk."""
    m = _CELL_RE.match(label)
    if not m:
        return None
    rowpart = m.group(1).upper()
    col = int(m.group(2)) - 1
    if len(rowpart) == 1:
        row = ord(rowpart) - ord("A")
    elif len(rowpart) == 2:
        row = (ord(rowpart[0]) - ord("A") + 1) * 26 + (ord(rowpart[1]) - ord("A"))
    else:
        return None
    if col < 0 or row < 0:
        return None
    return (row, col)


# --- Grid shape helpers ---


def grid_shape_for_image(
    image_size: tuple[int, int],
    short_axis_cells: int = 8,
) -> tuple[int, int]:
    """Pick (rows, cols) for an image, keeping cells approximately square.

    The shorter image axis gets `short_axis_cells` divisions; the longer axis
    gets a proportional count (rounded).
    """
    w, h = image_size
    if w <= h:
        cols = short_axis_cells
        rows = max(short_axis_cells, round(short_axis_cells * (h / w)))
    else:
        rows = short_axis_cells
        cols = max(short_axis_cells, round(short_axis_cells * (w / h)))
    return (int(rows), int(cols))


def cell_pixel_rect(
    row: int, col: int, rows: int, cols: int, image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Pixel bounds of a single cell: (x0, y0, x1, y1) inclusive-exclusive."""
    w, h = image_size
    cw = w / cols
    ch = h / rows
    x0 = int(round(col * cw))
    y0 = int(round(row * ch))
    x1 = int(round((col + 1) * cw))
    y1 = int(round((row + 1) * ch))
    return (x0, y0, x1, y1)


def cells_to_bbox(
    cell_labels: list[str],
    rows: int,
    cols: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return the bbox (x, y, w, h) spanning every valid cell label. None if none valid."""
    x0_list: list[int] = []
    y0_list: list[int] = []
    x1_list: list[int] = []
    y1_list: list[int] = []
    for label in cell_labels or []:
        rc = parse_cell(label)
        if rc is None:
            continue
        row, col = rc
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        x0, y0, x1, y1 = cell_pixel_rect(row, col, rows, cols, image_size)
        x0_list.append(x0)
        y0_list.append(y0)
        x1_list.append(x1)
        y1_list.append(y1)
    if not x0_list:
        return None
    x = min(x0_list)
    y = min(y0_list)
    w = max(x1_list) - x
    h = max(y1_list) - y
    return (x, y, w, h)


def cells_to_polygon(
    cell_labels: list[str],
    rows: int,
    cols: int,
    image_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return a polygon tracing the outer boundary of the union of picked cells.

    Uses a simple approach: for now returns the bbox rectangle of the union,
    since the rebuild is focused on getting position right first. For Phase 2
    (SAM) the precise boundary comes from the segmenter; this polygon is just
    the seed bbox.

    An L-shape or non-rectangular cell set currently gets represented as its
    enclosing rectangle. Good enough for Phase 1; can be replaced with a true
    cell-union outline later without API change.
    """
    bbox = cells_to_bbox(cell_labels, rows, cols, image_size)
    if bbox is None:
        return []
    x, y, w, h = bbox
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def cells_to_centroids(
    cell_labels: list[str],
    rows: int,
    cols: int,
    image_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return the pixel center of each valid cell, preserving input order."""
    out: list[tuple[int, int]] = []
    for label in cell_labels or []:
        rc = parse_cell(label)
        if rc is None:
            continue
        row, col = rc
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        x0, y0, x1, y1 = cell_pixel_rect(row, col, rows, cols, image_size)
        out.append(((x0 + x1) // 2, (y0 + y1) // 2))
    return out


# --- Drawing ---


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_label_grid(
    image_path: str,
    rows: int,
    cols: int,
    out_path: str,
) -> str:
    """Overlay a labeled grid on an image and save.

    Grid lines are thin white with a 1 px black shadow. Labels are row-letter
    + col-number centered in each cell, white with a black outline for
    legibility over any underlying imagery.
    """
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows

    for i in range(1, cols):
        x = int(i * cell_w)
        draw.line([(x + 1, 0), (x + 1, h)], fill=(0, 0, 0, 200), width=1)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 230), width=2)
    for j in range(1, rows):
        y = int(j * cell_h)
        draw.line([(0, y + 1), (w, y + 1)], fill=(0, 0, 0, 200), width=1)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 230), width=2)

    font_size = max(10, int(min(cell_w, cell_h) * 0.30))
    font = _load_font(font_size)

    for row in range(rows):
        for col in range(cols):
            label = f"{row_label(row)}{col + 1}"
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, fill="black", font=font)
            draw.text((tx, ty), label, fill="white", font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
