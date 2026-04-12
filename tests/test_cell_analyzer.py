"""Tests for the cell analyzer pipeline utilities."""

import pytest

from readwater.pipeline.cell_analyzer import _make_bbox, _make_cell_id


def test_make_bbox_center():
    bbox = _make_bbox((27.5, -82.5), 10.0)
    lat, lon = bbox.center
    assert abs(lat - 27.5) < 0.001
    assert abs(lon - (-82.5)) < 0.001


def test_make_bbox_size():
    bbox = _make_bbox((27.5, -82.5), 10.0)
    lat_span = bbox.north - bbox.south
    # 10 miles / 69 miles-per-degree ≈ 0.145 degrees
    assert abs(lat_span - 10 / 69.0) < 0.001


def test_make_cell_id_root():
    assert _make_cell_id(None, 0, 0) == "root"


def test_make_cell_id_child():
    assert _make_cell_id("root", 1, 2) == "root-1-2"


def test_make_cell_id_nested():
    assert _make_cell_id("root-1-2", 0, 1) == "root-1-2-0-1"
