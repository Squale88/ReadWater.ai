"""Tests for pipeline/structure/mosaic.py — tile selection, stitching, rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from readwater.api.providers.placeholder import PlaceholderProvider
from readwater.pipeline.structure.mosaic import (
    TILE_PX,
    Z16_CELL_PX,
    Mosaic,
    convex_hull,
    expand_plan,
    expand_polygon,
    polygon_iou,
    render_annotated,
    select_z18_centers,
)

CENTER = (27.5, -82.0)


# --- select_z18_centers ---


def test_tiny_anchor_yields_single_tile():
    bbox = (600, 600, 80, 80)  # 80 px in z16 is far smaller than one z18 tile.
    plan = select_z18_centers(bbox, CENTER)
    assert plan.rows == 1
    assert plan.cols == 1
    assert len(plan.centers) == 1
    assert plan.centers[0][0] == pytest.approx(CENTER, rel=1e-3)


def test_large_anchor_produces_multi_tile_grid():
    # A zoom-16 bbox spanning most of the image should need multiple z18 tiles.
    bbox = (100, 100, 1000, 1000)
    plan = select_z18_centers(bbox, CENTER)
    assert plan.rows >= 2
    assert plan.cols >= 2
    assert plan.rows <= 4
    assert plan.cols <= 4


def test_continuation_edges_extend_plan_within_cap():
    bbox = (400, 400, 400, 400)
    plan_no_ce = select_z18_centers(bbox, CENTER)
    plan_with_ce = select_z18_centers(
        bbox, CENTER,
        continuation_edges={"north": True, "east": True},
    )
    # Expansion only if there's room under MAX_GRID_CAP
    assert plan_with_ce.rows >= plan_no_ce.rows
    assert plan_with_ce.cols >= plan_no_ce.cols


def test_expand_plan_adds_row_on_north():
    bbox = (400, 400, 200, 200)
    plan = select_z18_centers(bbox, CENTER)
    anchor_center = plan.centers[plan.rows // 2][plan.cols // 2]
    expanded = expand_plan(plan, anchor_center, {"north": True})
    assert expanded.rows == plan.rows + 1
    assert expanded.cols == plan.cols


def test_expand_plan_respects_max_cap():
    # Force an already-large plan, then try to expand — should cap at 5.
    bbox = (0, 0, Z16_CELL_PX, Z16_CELL_PX)
    plan = select_z18_centers(
        bbox, CENTER,
        continuation_edges={"north": True, "south": True, "east": True, "west": True},
    )
    anchor_center = plan.centers[plan.rows // 2][plan.cols // 2]
    expanded = expand_plan(
        plan, anchor_center,
        {"north": True, "south": True, "east": True, "west": True},
    )
    assert expanded.rows <= 5
    assert expanded.cols <= 5


# --- Mosaic.build and pixel_to_latlon routing ---


async def test_mosaic_build_pastes_tiles_at_expected_origins(tmp_path: Path):
    provider = PlaceholderProvider(size=64, color=(10, 20, 30))
    bbox = (200, 200, 800, 800)
    plan = select_z18_centers(bbox, CENTER)

    mosaic = await Mosaic.build(
        plan, provider, tmp_path / "tiles", throttle_s=0.0,
    )
    # Mosaic canvas is capped at MAX_MOSAIC_DIM; width/height reflect rendered size.
    assert mosaic.width == pytest.approx(plan.cols * TILE_PX * mosaic.scale, abs=1)
    assert mosaic.height == pytest.approx(plan.rows * TILE_PX * mosaic.scale, abs=1)
    assert len(mosaic.tiles) == plan.rows * plan.cols

    # Tile origins are stored in nominal (pre-scale) space.
    for tp in mosaic.tiles:
        assert tp.origin_px == (tp.col * TILE_PX, tp.row * TILE_PX)


async def test_mosaic_pixel_to_latlon_routes_to_correct_tile(tmp_path: Path):
    provider = PlaceholderProvider(size=32)
    bbox = (100, 100, 1000, 1000)
    plan = select_z18_centers(bbox, CENTER)
    mosaic = await Mosaic.build(
        plan, provider, tmp_path / "tiles", throttle_s=0.0,
    )

    # Pixel at the center of tile (0, 0) should map near that tile's center lat/lon.
    # Pixel coordinates are in rendered-canvas space, so scale by mosaic.scale.
    tile_00 = [t for t in mosaic.tiles if t.row == 0 and t.col == 0][0]
    center_px = (TILE_PX // 2 * mosaic.scale, TILE_PX // 2 * mosaic.scale)
    lat, lon = mosaic.pixel_to_latlon(*center_px)
    assert lat == pytest.approx(tile_00.center_latlon[0], abs=1e-6)
    assert lon == pytest.approx(tile_00.center_latlon[1], abs=1e-6)


async def test_mosaic_tile_cache_prevents_refetch(tmp_path: Path):
    fetches = 0
    original_fetch = PlaceholderProvider.fetch

    async def counting_fetch(self, center, zoom, output_path, image_size=640):
        nonlocal fetches
        fetches += 1
        return await original_fetch(self, center, zoom, output_path, image_size)

    provider = PlaceholderProvider(size=16)
    # Monkeypatch for this test only
    PlaceholderProvider.fetch = counting_fetch  # type: ignore[method-assign]
    try:
        bbox = (400, 400, 400, 400)
        plan = select_z18_centers(bbox, CENTER)
        cache: dict = {}
        await Mosaic.build(plan, provider, tmp_path / "t1", throttle_s=0.0, tile_cache=cache)
        first = fetches
        await Mosaic.build(plan, provider, tmp_path / "t2", throttle_s=0.0, tile_cache=cache)
        second = fetches - first
        assert first > 0
        assert second == 0
    finally:
        PlaceholderProvider.fetch = original_fetch  # type: ignore[method-assign]


# --- Rendering ---


async def test_render_annotated_produces_image(tmp_path: Path):
    provider = PlaceholderProvider(size=16)
    bbox = (400, 400, 400, 400)
    plan = select_z18_centers(bbox, CENTER)
    mosaic = await Mosaic.build(plan, provider, tmp_path / "tiles", throttle_s=0.0)

    out_path = tmp_path / "annotated.png"
    render_annotated(
        base_image=mosaic.image,
        out_path=out_path,
        anchor_polygons_px=[("a1", [(100, 100), (500, 100), (500, 500), (100, 500)])],
        complex_polygons_px=[("cx", [(50, 50), (700, 50), (700, 700), (50, 700)])],
        influence_polygons_px=[("inf", [(0, 0), (1000, 0), (1000, 1000), (0, 1000)])],
        subzone_polygons_px=[("s1", [(200, 200), (300, 200), (300, 300), (200, 300)])],
    )
    assert out_path.exists()
    img = Image.open(out_path)
    assert img.size == (mosaic.width, mosaic.height)


# --- Geometry utilities ---


def test_convex_hull_of_rectangle_is_four_corners():
    hull = convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
    assert len(hull) == 4


def test_expand_polygon_enlarges_area():
    poly = [(100, 100), (200, 100), (200, 200), (100, 200)]
    expanded = expand_polygon(poly, 50)
    # Bounding box should grow
    xs_orig = [p[0] for p in poly]
    xs_exp = [p[0] for p in expanded]
    assert max(xs_exp) > max(xs_orig)
    assert min(xs_exp) < min(xs_orig)


def test_polygon_iou_identical_is_one():
    poly = [(10, 10), (50, 10), (50, 50), (10, 50)]
    iou = polygon_iou(poly, poly, (100, 100))
    assert iou == pytest.approx(1.0, abs=0.05)


def test_polygon_iou_disjoint_is_zero():
    a = [(10, 10), (30, 10), (30, 30), (10, 30)]
    b = [(70, 70), (90, 70), (90, 90), (70, 90)]
    iou = polygon_iou(a, b, (100, 100))
    assert iou == 0.0


# ------------------------------------------------------------------
# z18_tile_plan_from_latlon (TASK-5)
#
# At Rookery Bay (~26°N) one z18 scale=2 1280-px tile covers ~343 m on a
# side, so the asserts below are framed in those units. The 25% padding
# from the spec is exercised via the threshold tests.
# ------------------------------------------------------------------


def test_z18_tile_plan_returns_z18_fetch_plan_type():
    from readwater.models.structure import Z18FetchPlan
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    plan = z18_tile_plan_from_latlon((26.011172, -81.753546), 200.0)
    assert isinstance(plan, Z18FetchPlan)


def test_z18_tile_plan_tiny_extent_yields_one_tile():
    """A 10 m extent fits inside one tile (343 m), so the plan is 1×1."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    anchor = (26.011172, -81.753546)
    plan = z18_tile_plan_from_latlon(anchor, rough_extent_meters=10.0)
    assert len(plan.tile_centers) == 1
    # The single tile is centered on the anchor.
    assert plan.tile_centers[0] == pytest.approx(anchor, abs=1e-9)


def test_z18_tile_plan_700m_extent_yields_3x3():
    """700 m * 1.25 padding = 875 m needs >2 tiles per axis at lat 26
    (~343 m/tile), bumped to odd → 3 per axis → 9 total."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    plan = z18_tile_plan_from_latlon((26.011172, -81.753546), rough_extent_meters=700.0)
    assert len(plan.tile_centers) == 9
    # extent_meters covers (3 tiles * ~343 m) ≈ 1029 m, well past 875 m.
    assert plan.extent_meters > 875.0


def test_z18_tile_plan_centered_on_anchor():
    """For odd `n`, the middle tile of a row-major grid is centered on the
    anchor latlon. With n=3, the middle tile is index 4 (3*1 + 1 in 0-indexed)."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    anchor = (26.011172, -81.753546)
    plan = z18_tile_plan_from_latlon(anchor, rough_extent_meters=700.0)
    middle = plan.tile_centers[len(plan.tile_centers) // 2]
    assert middle == pytest.approx(anchor, abs=1e-9)


def test_z18_tile_plan_row_major_north_to_south_west_to_east():
    """Row 0 must be the northernmost (highest lat); within a row, col 0 is
    the westernmost (lowest lon)."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    plan = z18_tile_plan_from_latlon((26.011172, -81.753546), rough_extent_meters=700.0)
    # 3x3 → 9 centers, indices 0-8.
    first_row = plan.tile_centers[0:3]
    second_row = plan.tile_centers[3:6]
    third_row = plan.tile_centers[6:9]
    # All same row -> same lat; lat decreases from row 0 to row 2.
    assert first_row[0][0] == first_row[1][0] == first_row[2][0]
    assert first_row[0][0] > second_row[0][0] > third_row[0][0]
    # Within a row: col 0 lon < col 1 lon < col 2 lon.
    assert first_row[0][1] < first_row[1][1] < first_row[2][1]


def test_z18_tile_plan_honors_tile_budget():
    """tile_budget=9 caps total tiles at 9 even when extent demands more."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    # Crank the extent way past the budget (10 km > 5x5 tiles even at lat 26).
    plan = z18_tile_plan_from_latlon(
        (26.011172, -81.753546),
        rough_extent_meters=10_000.0,
        tile_budget=9,
    )
    assert len(plan.tile_centers) <= 9
    assert plan.tile_budget == 9


def test_z18_tile_plan_budget_clamps_to_largest_odd_square():
    """tile_budget=24 → max axis is floor(sqrt(24))=4, bumped down to odd 3
    (so 9 tiles, not 16). tile_budget=25 → axis 5, total 25."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    anchor = (26.011172, -81.753546)
    p24 = z18_tile_plan_from_latlon(anchor, rough_extent_meters=10_000.0, tile_budget=24)
    p25 = z18_tile_plan_from_latlon(anchor, rough_extent_meters=10_000.0, tile_budget=25)
    assert len(p24.tile_centers) == 9   # 3*3
    assert len(p25.tile_centers) == 25  # 5*5


def test_z18_tile_plan_zero_extent_still_yields_one_tile():
    """Defensive: a zero-extent anchor (e.g. coord-gen failure with tiny
    bbox) still gets a single-tile plan rather than an empty list."""
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    plan = z18_tile_plan_from_latlon((26.011172, -81.753546), rough_extent_meters=0.0)
    assert len(plan.tile_centers) == 1


def test_z18_tile_plan_invalid_budget_raises():
    from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

    with pytest.raises(ValueError):
        z18_tile_plan_from_latlon((26.0, -81.7), 100.0, tile_budget=0)


def test_z18_tile_plan_first_and_last_centers_within_tolerance():
    """Acceptance criterion sanity: first center is NW corner, last is SE,
    both at known offsets from the anchor."""
    from readwater.pipeline.structure.mosaic import (
        z18_tile_plan_from_latlon,
        z18_tile_span,
    )

    lat, lon = 26.011172, -81.753546
    plan = z18_tile_plan_from_latlon((lat, lon), rough_extent_meters=700.0)
    # n = 3, half = 1.0
    deg_lat_tile, deg_lon_tile = z18_tile_span(lat)
    expected_nw = (lat + 1.0 * deg_lat_tile, lon - 1.0 * deg_lon_tile)
    expected_se = (lat - 1.0 * deg_lat_tile, lon + 1.0 * deg_lon_tile)
    assert plan.tile_centers[0] == pytest.approx(expected_nw, abs=1e-9)
    assert plan.tile_centers[-1] == pytest.approx(expected_se, abs=1e-9)
