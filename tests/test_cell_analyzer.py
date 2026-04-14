"""Tests for cell analyzer coordinate geometry, ground coverage, and filenames.

Reference point: Marco Island, FL — center (25.94, -81.73).
All grid tests use sections=4 (4x4 = 16 sub-cells).
"""

import math

from readwater.pipeline.cell_analyzer import (
    MILES_PER_DEG_LAT,
    SECTIONS,
    _image_filename,
    _make_bbox,
    _make_cell_id,
    _miles_per_deg_lon,
    _role_for_zoom,
    _sub_cell_bbox,
    _subdivide_bbox,
    ground_coverage_miles,
)

MARCO = (25.94, -81.73)
MARCO_LAT = 25.94


# --- Longitude conversion ---


def test_miles_per_deg_lon_equator():
    assert abs(_miles_per_deg_lon(0.0) - MILES_PER_DEG_LAT) < 0.01


def test_miles_per_deg_lon_poles():
    assert abs(_miles_per_deg_lon(90.0)) < 0.01


def test_miles_per_deg_lon_marco_island():
    expected = MILES_PER_DEG_LAT * math.cos(math.radians(MARCO_LAT))
    assert abs(_miles_per_deg_lon(MARCO_LAT) - expected) < 0.001


# --- Ground coverage ---


def test_ground_coverage_zoom_10():
    cov = ground_coverage_miles(10, MARCO_LAT)
    # 2.5 * 24901 * cos(25.94) / 1024 ≈ 54.7
    assert 53 < cov < 56


def test_ground_coverage_zoom_12():
    cov = ground_coverage_miles(12, MARCO_LAT)
    assert 13 < cov < 15


def test_ground_coverage_zoom_14():
    cov = ground_coverage_miles(14, MARCO_LAT)
    assert 3 < cov < 4


def test_ground_coverage_zoom_16():
    cov = ground_coverage_miles(16, MARCO_LAT)
    assert 0.8 < cov < 1.0


def test_ground_coverage_zoom_18():
    cov = ground_coverage_miles(18, MARCO_LAT)
    assert 0.2 < cov < 0.25


def test_ground_coverage_monotonically_decreasing():
    zooms = [10, 12, 14, 16, 18]
    coverages = [ground_coverage_miles(z, MARCO_LAT) for z in zooms]
    for i in range(len(coverages) - 1):
        assert coverages[i] > coverages[i + 1]


# --- Bounding box ---


def test_bbox_center_matches_input():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lat, lon = bbox.center
    assert abs(lat - MARCO[0]) < 1e-10
    assert abs(lon - MARCO[1]) < 1e-10


def test_bbox_lat_span_matches_size():
    size = ground_coverage_miles(12, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lat_span_miles = (bbox.north - bbox.south) * MILES_PER_DEG_LAT
    assert abs(lat_span_miles - size) < 0.01


def test_bbox_lon_span_matches_size():
    size = ground_coverage_miles(12, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lon_span_miles = (bbox.east - bbox.west) * _miles_per_deg_lon(MARCO_LAT)
    assert abs(lon_span_miles - size) < 0.01


def test_bbox_is_wider_in_degrees_than_tall():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert (bbox.east - bbox.west) > (bbox.north - bbox.south)


def test_bbox_north_south_symmetric():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert abs((bbox.north - MARCO[0]) - (MARCO[0] - bbox.south)) < 1e-10


def test_bbox_east_west_symmetric():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert abs((bbox.east - MARCO[1]) - (MARCO[1] - bbox.west)) < 1e-10


# --- 4x4 grid subdivision ---


def _marco_bbox():
    size = ground_coverage_miles(10, MARCO_LAT)
    return _make_bbox(MARCO, size)


def test_subdivide_4x4_returns_16():
    assert len(_subdivide_bbox(_marco_bbox(), SECTIONS)) == 16


def test_subdivide_4x4_row_col_coverage():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    positions = {(r, c) for r, c, _ in subs}
    expected = {(r, c) for r in range(4) for c in range(4)}
    assert positions == expected


def test_subdivide_4x4_centers_within_parent():
    bbox = _marco_bbox()
    for row, col, (lat, lon) in _subdivide_bbox(bbox, SECTIONS):
        assert bbox.south < lat < bbox.north
        assert bbox.west < lon < bbox.east


def test_subdivide_4x4_centers_ordered_north_to_south():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    by_row = {}
    for row, col, (lat, _) in subs:
        by_row.setdefault(row, []).append(lat)
    for row in range(SECTIONS - 1):
        assert min(by_row[row]) > max(by_row[row + 1])


def test_subdivide_4x4_centers_ordered_west_to_east():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    by_col = {}
    for row, col, (_, lon) in subs:
        by_col.setdefault(col, []).append(lon)
    for col in range(SECTIONS - 1):
        assert max(by_col[col]) < min(by_col[col + 1])


def test_subdivide_4x4_centers_evenly_spaced_lat():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    lats = sorted({lat for _, _, (lat, _) in subs}, reverse=True)
    assert len(lats) == SECTIONS
    spacing = lats[0] - lats[1]
    for i in range(len(lats) - 1):
        assert abs((lats[i] - lats[i + 1]) - spacing) < 1e-10


def test_subdivide_4x4_centers_evenly_spaced_lon():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    lons = sorted({lon for _, _, (_, lon) in subs})
    assert len(lons) == SECTIONS
    spacing = lons[1] - lons[0]
    for i in range(len(lons) - 1):
        assert abs((lons[i + 1] - lons[i]) - spacing) < 1e-10


# --- Sub-cell bounding boxes: tiling ---


def test_sub_cell_4x4_bboxes_tile_parent():
    bbox = _marco_bbox()
    norths, souths, easts, wests = [], [], [], []
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            norths.append(sb.north)
            souths.append(sb.south)
            easts.append(sb.east)
            wests.append(sb.west)
    assert abs(max(norths) - bbox.north) < 1e-10
    assert abs(min(souths) - bbox.south) < 1e-10
    assert abs(max(easts) - bbox.east) < 1e-10
    assert abs(min(wests) - bbox.west) < 1e-10


def test_sub_cell_4x4_no_gaps_horizontal():
    bbox = _marco_bbox()
    for row in range(SECTIONS):
        for col in range(SECTIONS - 1):
            left = _sub_cell_bbox(bbox, row, col, SECTIONS)
            right = _sub_cell_bbox(bbox, row, col + 1, SECTIONS)
            assert abs(left.east - right.west) < 1e-10


def test_sub_cell_4x4_no_gaps_vertical():
    bbox = _marco_bbox()
    for col in range(SECTIONS):
        for row in range(SECTIONS - 1):
            upper = _sub_cell_bbox(bbox, row, col, SECTIONS)
            lower = _sub_cell_bbox(bbox, row + 1, col, SECTIONS)
            assert abs(upper.south - lower.north) < 1e-10


def test_sub_cell_4x4_center_matches_subdivide():
    bbox = _marco_bbox()
    subs = {(r, c): center for r, c, center in _subdivide_bbox(bbox, SECTIONS)}
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            assert abs(sb.center[0] - subs[(row, col)][0]) < 1e-10
            assert abs(sb.center[1] - subs[(row, col)][1]) < 1e-10


def test_sub_cell_4x4_size_is_quarter_of_parent():
    bbox = _marco_bbox()
    parent_h = bbox.north - bbox.south
    parent_w = bbox.east - bbox.west
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            assert abs((sb.north - sb.south) - parent_h / SECTIONS) < 1e-10
            assert abs((sb.east - sb.west) - parent_w / SECTIONS) < 1e-10


def test_two_level_4x4_subdivision_tiles():
    """Subdivide a sub-cell again — 16 grandchild bboxes should tile it."""
    parent = _marco_bbox()
    child = _sub_cell_bbox(parent, 1, 2, SECTIONS)
    for row in range(SECTIONS):
        for col in range(SECTIONS - 1):
            left = _sub_cell_bbox(child, row, col, SECTIONS)
            right = _sub_cell_bbox(child, row, col + 1, SECTIONS)
            assert abs(left.east - right.west) < 1e-10
    for col in range(SECTIONS):
        for row in range(SECTIONS - 1):
            upper = _sub_cell_bbox(child, row, col, SECTIONS)
            lower = _sub_cell_bbox(child, row + 1, col, SECTIONS)
            assert abs(upper.south - lower.north) < 1e-10


# --- Cell ID generation ---


def test_make_cell_id_root():
    assert _make_cell_id(None, 0, 0) == "root"


def test_make_cell_id_child():
    # row 1, col 2 in 4x4 grid = cell 7
    assert _make_cell_id("root", 1, 2) == "root-7"


def test_make_cell_id_nested():
    # row 0, col 1 in 4x4 grid = cell 2
    assert _make_cell_id("root-7", 0, 1) == "root-7-2"


# --- Image filename ---


def test_image_filename_root():
    assert _image_filename("root", 0) == "z0.png"


def test_image_filename_root_with_provider():
    assert _image_filename("root", 0, "google_static") == "z0_google_static.png"


def test_image_filename_depth1():
    assert _image_filename("root-14", 1) == "z0_14.png"


def test_image_filename_depth2_with_provider():
    assert _image_filename("root-14-3", 2, "naip") == "z0_14_3_naip.png"


def test_image_filename_cell_16():
    """Cell 16 (row 3, col 3) is valid in a 4x4 grid."""
    assert _image_filename("root-16", 1) == "z0_16.png"


# --- Role for zoom ---


def test_role_overview_at_low_zoom():
    for z in [10, 12, 14, 15]:
        assert _role_for_zoom(z) == "overview"


def test_role_structure_at_high_zoom():
    for z in [16, 18, 20]:
        assert _role_for_zoom(z) == "structure"
