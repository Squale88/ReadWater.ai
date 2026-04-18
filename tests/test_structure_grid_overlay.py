"""Tests for pipeline/structure/grid_overlay.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from readwater.pipeline.structure.grid_overlay import (
    cell_pixel_rect,
    cells_to_bbox,
    cells_to_centroids,
    cells_to_polygon,
    draw_label_grid,
    grid_shape_for_image,
    parse_cell,
    row_label,
)


# --- row_label ---


def test_row_label_single_letters():
    assert row_label(0) == "A"
    assert row_label(1) == "B"
    assert row_label(25) == "Z"


def test_row_label_two_letters():
    assert row_label(26) == "AA"
    assert row_label(27) == "AB"
    assert row_label(51) == "AZ"
    assert row_label(52) == "BA"


# --- parse_cell ---


def test_parse_cell_basic():
    assert parse_cell("A1") == (0, 0)
    assert parse_cell("H8") == (7, 7)


def test_parse_cell_lowercase_and_whitespace():
    assert parse_cell("a1") == (0, 0)
    assert parse_cell("  c3  ") == (2, 2)


def test_parse_cell_two_letter_row():
    assert parse_cell("AA1") == (26, 0)
    assert parse_cell("AB2") == (27, 1)


def test_parse_cell_invalid():
    assert parse_cell("") is None
    assert parse_cell("1A") is None
    assert parse_cell("A") is None
    assert parse_cell("A0") == (0, -1) or parse_cell("A0") is None
    # By our implementation, "A0" parses to (0, -1) but that negative col is
    # caught by the `col < 0` guard and returns None.
    assert parse_cell("A0") is None


# --- grid_shape_for_image ---


def test_grid_shape_square_image():
    rows, cols = grid_shape_for_image((1600, 1600), short_axis_cells=8)
    assert rows == 8
    assert cols == 8


def test_grid_shape_wide_image():
    # 1600 wide x 800 tall -> rows=8 (short=height), cols proportional.
    rows, cols = grid_shape_for_image((1600, 800), short_axis_cells=8)
    assert rows == 8
    assert cols >= 8  # wider than tall
    assert cols == pytest.approx(16, abs=1)


def test_grid_shape_tall_image():
    # 960 wide x 1600 tall -> cols=8 (short=width), rows proportional.
    rows, cols = grid_shape_for_image((960, 1600), short_axis_cells=8)
    assert cols == 8
    assert rows >= 8
    assert rows == pytest.approx(13, abs=1)


# --- cell_pixel_rect ---


def test_cell_pixel_rect_basic():
    # 8x8 grid on a 1600x1600 image; each cell is 200x200.
    rect = cell_pixel_rect(0, 0, 8, 8, (1600, 1600))
    assert rect == (0, 0, 200, 200)
    rect = cell_pixel_rect(7, 7, 8, 8, (1600, 1600))
    assert rect == (1400, 1400, 1600, 1600)


# --- cells_to_bbox ---


def test_cells_to_bbox_single_cell():
    bbox = cells_to_bbox(["A1"], 8, 8, (1600, 1600))
    assert bbox == (0, 0, 200, 200)


def test_cells_to_bbox_rectangle_union():
    bbox = cells_to_bbox(["C3", "C4", "D3", "D4"], 8, 8, (1600, 1600))
    # C3 = row 2 col 2, D4 = row 3 col 3.  → x 2*200=400 to 4*200=800, y 2*200=400 to 4*200=800
    assert bbox == (400, 400, 400, 400)


def test_cells_to_bbox_ignores_invalid_labels():
    bbox = cells_to_bbox(["A1", "junk", "B2"], 8, 8, (1600, 1600))
    # Valid cells: A1 and B2. A1 = row 0 col 0, B2 = row 1 col 1.
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 0 and y == 0
    assert w == 400 and h == 400


def test_cells_to_bbox_ignores_out_of_bounds():
    bbox = cells_to_bbox(["A1", "Z99"], 8, 8, (1600, 1600))
    assert bbox == (0, 0, 200, 200)


def test_cells_to_bbox_empty_returns_none():
    assert cells_to_bbox([], 8, 8, (1600, 1600)) is None
    assert cells_to_bbox(["junk"], 8, 8, (1600, 1600)) is None


# --- cells_to_polygon ---


def test_cells_to_polygon_is_rectangle():
    poly = cells_to_polygon(["C3", "C4", "D3", "D4"], 8, 8, (1600, 1600))
    assert len(poly) == 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == 400 and max(xs) == 800
    assert min(ys) == 400 and max(ys) == 800


# --- cells_to_centroids ---


def test_cells_to_centroids_preserves_order():
    cs = cells_to_centroids(["A1", "H8"], 8, 8, (1600, 1600))
    assert cs[0] == (100, 100)
    assert cs[1] == (1500, 1500)


# --- draw_label_grid ---


def test_draw_label_grid_writes_file(tmp_path: Path):
    src = tmp_path / "src.png"
    Image.new("RGB", (400, 400), (128, 128, 128)).save(src)
    out = tmp_path / "out.png"
    path = draw_label_grid(str(src), rows=4, cols=4, out_path=str(out))
    assert path == str(out)
    assert out.exists()
    img = Image.open(out)
    assert img.size == (400, 400)
