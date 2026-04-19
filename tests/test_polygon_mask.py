"""Tests for the generic polygon mask pipeline (rasterize + overlay)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import rasterio  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False

from readwater.pipeline.polygon_mask import (
    resample_bool_mask,
    save_polygon_overlay_png,
)

pytestmark_cv = pytest.mark.skipif(not HAS_CV, reason="requires 'cv' extras")

TINT_RED = (255, 0, 0, 200)


# --- Overlay alignment (pure numpy, no geo deps needed) ---


def test_overlay_identical_bboxes(tmp_path: Path):
    """If mask and base cover the same bbox, overlay should preserve mask."""
    base = Image.new("RGB", (40, 40), (80, 120, 160))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_polygon_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
        rgba=TINT_RED,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    tinted_mean = arr[10:20, 10:20].mean(axis=(0, 1))
    untinted_mean = arr[0:10, 0:10].mean(axis=(0, 1))
    assert tinted_mean[0] > untinted_mean[0] + 30


def test_overlay_mask_resizes_to_base(tmp_path: Path):
    """Mask at different resolution than base: overlay must resize cleanly."""
    base = Image.new("RGB", (80, 80), (100, 100, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_polygon_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
        rgba=TINT_RED,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    assert arr.shape == (80, 80, 3)
    assert arr[40, 40, 0] > arr[5, 5, 0] + 20


def test_overlay_different_bboxes(tmp_path: Path):
    """Mask covers only the east half of the base's bbox."""
    base = Image.new("RGB", (100, 100), (80, 80, 80))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    base_bbox = (-82.0, 26.0, -81.9, 26.1)
    mask_bbox = (-81.95, 26.0, -81.9, 26.1)

    mask = np.ones((50, 50), dtype=bool)

    out = tmp_path / "overlay.png"
    save_polygon_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=base_bbox,
        mask_bbox_4326=mask_bbox,
        out_path=out,
        rgba=TINT_RED,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    west_red = arr[:, 0:45, 0].mean()
    east_red = arr[:, 55:, 0].mean()
    assert east_red > west_red + 20


def test_overlay_outline_only(tmp_path: Path):
    base = Image.new("RGB", (60, 60), (100, 100, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_polygon_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
        rgba=TINT_RED,
        outline_only=True,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    interior = arr[30, 30, 0]
    boundary = arr[20, 30, 0]
    assert boundary > interior + 20


# --- resample_bool_mask ---


def test_resample_matching_bbox_and_size_is_noop():
    mask = np.array([[True, False], [False, True]], dtype=bool)
    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = resample_bool_mask(mask, bbox, bbox, (2, 2))
    assert np.array_equal(out, mask)


def test_resample_nearest_neighbor_scales_up():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:, 5:] = True
    bbox = (0.0, 0.0, 1.0, 1.0)
    out = resample_bool_mask(mask, bbox, bbox, (100, 100))
    assert out.shape == (100, 100)
    # The True quadrant should still be the bottom-right half-ish.
    assert out[80, 80]
    assert not out[20, 20]


# --- Rasterize (requires rasterio/shapely) ---


@pytestmark_cv
def test_rasterize_single_polygon(tmp_path: Path):
    from readwater.pipeline.polygon_mask import rasterize_polygons

    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-81.95, 26.0], [-81.9, 26.0],
                        [-81.9, 26.1], [-81.95, 26.1], [-81.95, 26.0],
                    ]],
                },
                "properties": {},
            }
        ],
    }
    gj = tmp_path / "one.geojson"
    gj.write_text(json.dumps(geo))

    result = rasterize_polygons(
        geojson_path=gj,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(100, 100),
        out_mask_png=tmp_path / "mask.png",
        out_mask_tif=tmp_path / "mask.tif",
    )
    assert 0.4 < result.covered_fraction < 0.6
    assert Path(result.mask_path).exists()
    assert Path(result.tif_path).exists()


@pytestmark_cv
def test_rasterize_empty_geojson(tmp_path: Path):
    from readwater.pipeline.polygon_mask import rasterize_polygons

    gj = tmp_path / "empty.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    result = rasterize_polygons(
        geojson_path=gj,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(50, 50),
        out_mask_png=tmp_path / "mask.png",
    )
    assert result.covered_fraction == 0.0


@pytestmark_cv
def test_rasterize_multiple_polygons(tmp_path: Path):
    from readwater.pipeline.polygon_mask import rasterize_polygons

    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-82.0, 26.0], [-81.99, 26.0],
                        [-81.99, 26.01], [-82.0, 26.01], [-82.0, 26.0],
                    ]],
                },
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-81.91, 26.09], [-81.90, 26.09],
                        [-81.90, 26.10], [-81.91, 26.10], [-81.91, 26.09],
                    ]],
                },
                "properties": {},
            },
        ],
    }
    gj = tmp_path / "multi.geojson"
    gj.write_text(json.dumps(geo))

    result = rasterize_polygons(
        geojson_path=gj,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(100, 100),
        out_mask_png=tmp_path / "mask.png",
    )
    assert 0 < result.covered_fraction < 0.1
