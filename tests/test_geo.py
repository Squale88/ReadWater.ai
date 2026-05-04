"""Tests for pixel <-> lat/lon helpers in pipeline/geo.py."""

from __future__ import annotations

import math

import pytest

from readwater.pipeline.cell_analyzer import ground_coverage_miles
from readwater.pipeline.geo import (
    clip_polygon_to_rect,
    deg_lat_per_pixel,
    deg_lon_per_pixel,
    latlon_to_pixel,
    meters_per_pixel,
    pixel_to_latlon,
    polygon_px_to_latlon,
)

# Continental US coastal latitudes we actually care about, plus bracketing points.
LATITUDES = [0.0, 27.5, 45.0, 60.0]
IMAGE_SIZE_PX = 1280
TILE_PX = 1280


# --- Round-trip correctness ---


@pytest.mark.parametrize("lat", LATITUDES)
@pytest.mark.parametrize("zoom", [14, 16, 18])
@pytest.mark.parametrize("px,py", [(0, 0), (640, 640), (1279, 1279), (100, 900)])
def test_pixel_roundtrip_at_various_latitudes(lat: float, zoom: int, px: int, py: int):
    center_lat = lat
    center_lon = -81.0
    geo = pixel_to_latlon(px, py, IMAGE_SIZE_PX, center_lat, center_lon, zoom)
    back_x, back_y = latlon_to_pixel(
        geo[0], geo[1], IMAGE_SIZE_PX, center_lat, center_lon, zoom,
    )
    assert back_x == pytest.approx(px, abs=1e-6)
    assert back_y == pytest.approx(py, abs=1e-6)


def test_center_pixel_maps_to_center_latlon():
    lat, lon = pixel_to_latlon(640, 640, 1280, 27.5, -82.0, 18)
    assert lat == pytest.approx(27.5, abs=1e-9)
    assert lon == pytest.approx(-82.0, abs=1e-9)


def test_pixel_y_increases_southward():
    north = pixel_to_latlon(640, 100, 1280, 27.5, -82.0, 18)[0]
    south = pixel_to_latlon(640, 1100, 1280, 27.5, -82.0, 18)[0]
    assert north > south


def test_pixel_x_increases_eastward():
    west = pixel_to_latlon(100, 640, 1280, 27.5, -82.0, 18)[1]
    east = pixel_to_latlon(1100, 640, 1280, 27.5, -82.0, 18)[1]
    assert east > west


# --- Georeferencing against independently computed tile span ---


@pytest.mark.parametrize("lat", [27.5, 45.0])
@pytest.mark.parametrize("zoom", [16, 18])
def test_nw_corner_matches_ground_coverage_formula(lat: float, zoom: int):
    """pixel (0,0) should sit at the NW corner of the tile's ground footprint.

    This catches direction-symmetric bugs that pure round-trip tests miss.
    """
    center_lat = lat
    center_lon = -82.0
    # Independent derivation of the tile's half-span from ground_coverage_miles.
    cov_miles = ground_coverage_miles(zoom, center_lat, image_size=640)
    half_span_miles = cov_miles / 2.0
    # 1 deg lat ~= 69 miles; longitude compressed by cos(lat).
    half_span_deg_lat = half_span_miles / 69.0
    half_span_deg_lon = half_span_deg_lat / math.cos(math.radians(center_lat))
    expected_nw_lat = center_lat + half_span_deg_lat
    expected_nw_lon = center_lon - half_span_deg_lon

    nw_lat, nw_lon = pixel_to_latlon(0, 0, 1280, center_lat, center_lon, zoom)
    # Mercator-vs-great-circle mismatch at high-lat is small; allow 0.5% slop.
    rel_tol_lat = abs(expected_nw_lat - center_lat) * 0.005
    rel_tol_lon = abs(expected_nw_lon - center_lon) * 0.005
    assert nw_lat == pytest.approx(expected_nw_lat, abs=max(rel_tol_lat, 1e-5))
    assert nw_lon == pytest.approx(expected_nw_lon, abs=max(rel_tol_lon, 1e-5))


# --- Scale behavior ---


def test_scale_2_doubles_pixel_density():
    # Two pixels at scale=2 should cover the same ground as one pixel at scale=1.
    scale1 = meters_per_pixel(16, 27.5, scale=1)
    scale2 = meters_per_pixel(16, 27.5, scale=2)
    assert scale1 == pytest.approx(scale2 * 2, rel=1e-9)


def test_meters_per_pixel_halves_per_zoom():
    a = meters_per_pixel(16, 27.5)
    b = meters_per_pixel(17, 27.5)
    assert b == pytest.approx(a / 2, rel=1e-9)


def test_deg_lon_larger_than_deg_lat_except_equator():
    # At nonzero latitude, cos(lat) < 1 means deg_lon > deg_lat per pixel.
    assert deg_lon_per_pixel(18, 45.0) > deg_lat_per_pixel(18, 45.0)
    # At the equator they are equal.
    assert deg_lon_per_pixel(18, 0.0) == pytest.approx(deg_lat_per_pixel(18, 0.0))


# --- Polygon conversion ---


def test_polygon_px_to_latlon_preserves_length_and_order():
    poly = [(100, 100), (1100, 100), (1100, 1100), (100, 1100)]
    latlon = polygon_px_to_latlon(poly, 1280, 27.5, -82.0, 18)
    assert len(latlon) == len(poly)
    # Vertex ordering: first is NW => highest lat, westernmost lon
    # Last two: south-east corner should be the third entry.
    assert latlon[0][0] > latlon[2][0]  # NW lat > SE lat
    assert latlon[0][1] < latlon[2][1]  # NW lon < SE lon


# --- Clipping ---


def test_clip_polygon_inside_returns_self():
    poly = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)]
    clipped = clip_polygon_to_rect(poly, 500, 500)
    assert len(clipped) == 4


def test_clip_polygon_partial_clip_produces_valid_polygon():
    # Triangle spilling past the right edge of a 100x100 canvas.
    poly = [(50.0, 50.0), (150.0, 50.0), (100.0, 150.0)]
    clipped = clip_polygon_to_rect(poly, 100, 100)
    assert len(clipped) >= 3
    for x, y in clipped:
        assert 0 <= x <= 100
        assert 0 <= y <= 100


def test_clip_polygon_fully_outside_returns_empty():
    poly = [(200.0, 200.0), (300.0, 200.0), (250.0, 300.0)]
    assert clip_polygon_to_rect(poly, 100, 100) == []


def test_clip_polygon_collapsed_returns_empty():
    # Degenerate line.
    poly = [(10.0, 10.0), (20.0, 20.0)]
    assert clip_polygon_to_rect(poly, 100, 100) == []
