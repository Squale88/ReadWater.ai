"""Tests for ClickBoxExtractor (four modes) + feature-type → mode routing."""

from __future__ import annotations

import pytest
from PIL import Image

from readwater.pipeline.structure.extractors import (
    ClickBoxExtractor,
    STRUCTURE_TYPE_TO_MODE,
    SUBZONE_TYPE_TO_MODE,
    get_extractor,
    is_subzone_type_allowed,
    mode_for,
)


@pytest.fixture
def canvas():
    return Image.new("RGB", (1000, 1000), (50, 100, 150))


# --- region mode ---


def test_region_single_positive_produces_rect(canvas):
    ex = ClickBoxExtractor("region")
    out = ex.extract(canvas, positive_points=[(500, 500)], negative_points=[])
    poly = out.pixel_polygon
    assert len(poly) == 4
    assert out.extractor_name == "clickbox"
    # Rectangle is padded by 5% of 1000 = 50 px around the single point.
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert max(xs) - min(xs) == pytest.approx(100, abs=2)
    assert max(ys) - min(ys) == pytest.approx(100, abs=2)


def test_region_multiple_positives_spans_all(canvas):
    ex = ClickBoxExtractor("region")
    out = ex.extract(
        canvas,
        positive_points=[(200, 300), (700, 800)],
        negative_points=[],
    )
    poly = out.pixel_polygon
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # With 5% pad = 50 px, bbox should span roughly 150..750 and 250..850.
    assert min(xs) == pytest.approx(150, abs=2)
    assert max(xs) == pytest.approx(750, abs=2)
    assert min(ys) == pytest.approx(250, abs=2)
    assert max(ys) == pytest.approx(850, abs=2)


def test_region_respects_bbox_hint(canvas):
    ex = ClickBoxExtractor("region")
    out = ex.extract(
        canvas,
        positive_points=[(500, 500)],
        negative_points=[],
        bbox_hint=(400, 400, 200, 200),  # [400,600] x [400,600]
    )
    for x, y in out.pixel_polygon:
        assert 400 <= x <= 600
        assert 400 <= y <= 600


def test_region_empty_positives_returns_empty(canvas):
    ex = ClickBoxExtractor("region")
    out = ex.extract(canvas, positive_points=[], negative_points=[])
    assert out.pixel_polygon == []


# --- corridor mode ---


def test_corridor_two_positives_produces_thick_line_rect(canvas):
    ex = ClickBoxExtractor("corridor")
    out = ex.extract(
        canvas,
        positive_points=[(200, 500), (800, 500)],
        negative_points=[],
    )
    poly = out.pixel_polygon
    assert len(poly) == 4
    # Horizontal corridor: width is 3% of 1000 = 30 px; polygon spans 200..800 in x.
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == pytest.approx(200, abs=2)
    assert max(xs) == pytest.approx(800, abs=2)
    # y spans roughly 485..515
    assert max(ys) - min(ys) == pytest.approx(30, abs=2)


def test_corridor_single_positive_falls_through_to_point(canvas):
    ex = ClickBoxExtractor("corridor")
    out = ex.extract(
        canvas,
        positive_points=[(500, 500)],
        negative_points=[],
    )
    # Corridor with one point degenerates to point_feature behavior (small square).
    poly = out.pixel_polygon
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # point_side = 4% of 1000 = 40 px
    assert max(xs) - min(xs) == pytest.approx(40, abs=2)
    assert max(ys) - min(ys) == pytest.approx(40, abs=2)


# --- point_feature mode ---


def test_point_feature_produces_small_square(canvas):
    ex = ClickBoxExtractor("point_feature")
    out = ex.extract(
        canvas,
        positive_points=[(500, 500)],
        negative_points=[],
    )
    poly = out.pixel_polygon
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # side = 4% of 1000 = 40
    assert max(xs) - min(xs) == pytest.approx(40, abs=2)
    assert max(ys) - min(ys) == pytest.approx(40, abs=2)


# --- edge_band mode ---


def test_edge_band_produces_thin_rectangle_along_line(canvas):
    ex = ClickBoxExtractor("edge_band")
    out = ex.extract(
        canvas,
        positive_points=[(100, 500), (900, 500)],
        negative_points=[],
    )
    poly = out.pixel_polygon
    assert len(poly) == 4
    ys = [p[1] for p in poly]
    # Edge band width = 2% of 1000 = 20
    assert max(ys) - min(ys) == pytest.approx(20, abs=2)


# --- Unknown mode ---


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        ClickBoxExtractor("nonsense")


# --- Routing table ---


def test_every_anchor_type_maps_to_a_mode():
    valid_modes = {"region", "corridor", "point_feature", "edge_band"}
    for structure_type, mode in STRUCTURE_TYPE_TO_MODE.items():
        assert mode in valid_modes, f"{structure_type} routes to invalid {mode}"


def test_every_subzone_type_maps_to_a_mode():
    valid_modes = {"region", "corridor", "point_feature", "edge_band"}
    for sz_type, mode in SUBZONE_TYPE_TO_MODE.items():
        assert mode in valid_modes, f"{sz_type} routes to invalid {mode}"


def test_v1_subzone_whitelist_size_is_five():
    assert len(SUBZONE_TYPE_TO_MODE) == 5


def test_is_subzone_type_allowed():
    assert is_subzone_type_allowed("drain_throat")
    assert is_subzone_type_allowed("point_tip")
    assert not is_subzone_type_allowed("receiving_basin_lane")
    assert not is_subzone_type_allowed("nonsense")


def test_mode_for_anchor_types():
    assert mode_for("drain", "anchor") == "corridor"
    assert mode_for("oyster_bar", "anchor") == "region"
    assert mode_for("island_edge", "anchor") == "edge_band"


def test_mode_for_subzone_types():
    assert mode_for("drain_throat", "subzone") == "corridor"
    assert mode_for("point_tip", "subzone") == "point_feature"
    assert mode_for("oyster_bar_edge", "subzone") == "edge_band"


def test_mode_for_unknown_subzone_uses_default_region():
    assert mode_for("receiving_basin_lane", "subzone") == "region"


def test_mode_for_complex_member_vocabulary():
    # Complex members use feature_type values like basin, channel, etc.
    assert mode_for("basin", "anchor") == "region"
    assert mode_for("channel", "anchor") == "corridor"
    assert mode_for("shoreline", "anchor") == "edge_band"


# --- get_extractor dispatch ---


def test_get_extractor_returns_correct_mode_instance():
    reg_ex = get_extractor("region")
    corr_ex = get_extractor("corridor")
    assert reg_ex.mode == "region"
    assert corr_ex.mode == "corridor"


def test_get_extractor_unknown_mode_falls_back_to_region():
    ex = get_extractor("not_a_mode")
    assert ex.mode == "region"


def test_get_extractor_accepts_custom_registry():
    custom = {"region": ClickBoxExtractor("point_feature")}  # deliberately swapped
    ex = get_extractor("region", registry=custom)
    assert ex.mode == "point_feature"
