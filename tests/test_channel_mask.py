"""Tests for channel mask rasterization and overlay alignment."""

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

from readwater.pipeline.channel_mask import save_channel_overlay_png

pytestmark_cv = pytest.mark.skipif(not HAS_CV, reason="requires 'cv' extras")


# --- Overlay alignment (pure numpy, no geo deps needed) ---


def test_channel_overlay_identical_bboxes(tmp_path: Path):
    """If mask and base cover the same bbox, overlay should preserve mask."""
    base = Image.new("RGB", (40, 40), (80, 120, 160))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True  # 10x10 channel in the middle

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_channel_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
        channel_rgba=(255, 0, 0, 200),
    )

    arr = np.array(Image.open(out).convert("RGB"))
    # The tinted region should have more red than untinted.
    tinted_mean = arr[10:20, 10:20].mean(axis=(0, 1))
    untinted_mean = arr[0:10, 0:10].mean(axis=(0, 1))
    assert tinted_mean[0] > untinted_mean[0] + 30


def test_channel_overlay_mask_resizes_to_base(tmp_path: Path):
    """Mask at different resolution than base: overlay must resize cleanly."""
    base = Image.new("RGB", (80, 80), (100, 100, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    # 20x20 mask, bbox matches base.
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True  # center block

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_channel_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    assert arr.shape == (80, 80, 3)
    # Center of output should be tinted; corner should not.
    assert arr[40, 40, 0] > arr[5, 5, 0] + 20


def test_channel_overlay_different_bboxes(tmp_path: Path):
    """Mask covers only a sub-area of base. Overlay should align correctly."""
    base = Image.new("RGB", (100, 100), (80, 80, 80))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    # Base covers a 0.1 x 0.1 deg tile; mask covers only the east half.
    base_bbox = (-82.0, 26.0, -81.9, 26.1)
    mask_bbox = (-81.95, 26.0, -81.9, 26.1)

    mask = np.ones((50, 50), dtype=bool)  # all True in mask's (east-half) area

    out = tmp_path / "overlay.png"
    save_channel_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=base_bbox,
        mask_bbox_4326=mask_bbox,
        out_path=out,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    # West half (base.x in [0, 50)): lat/lon x in [-82.0, -81.95] — outside
    # mask bbox; should be untinted.
    west_red = arr[:, 0:45, 0].mean()
    # East half (x in [50, 100)): lat/lon x in [-81.95, -81.9] — inside mask.
    east_red = arr[:, 55:, 0].mean()
    assert east_red > west_red + 20


def test_channel_overlay_outline_only(tmp_path: Path):
    """outline_only should paint only boundary pixels, not the full fill."""
    base = Image.new("RGB", (60, 60), (100, 100, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True  # 20x20 block

    bbox = (-82.0, 26.0, -81.9, 26.1)
    out = tmp_path / "overlay.png"
    save_channel_overlay_png(
        rgb_image_path=base_path,
        mask_bool=mask,
        rgb_bbox_4326=bbox,
        mask_bbox_4326=bbox,
        out_path=out,
        outline_only=True,
    )

    arr = np.array(Image.open(out).convert("RGB"))
    # Interior pixel (30, 30) should NOT be tinted heavily — only outline is.
    interior = arr[30, 30, 0]
    # A boundary pixel should be tinted.
    boundary = arr[20, 30, 0]
    assert boundary > interior + 20


# --- Rasterization (requires rasterio/shapely via cv extras) ---


@pytestmark_cv
def test_rasterize_channels_from_synthetic_geojson(tmp_path: Path):
    from readwater.pipeline.channel_mask import rasterize_channels

    # Channel polygon covering the right half of our bbox.
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-81.95, 26.0],
                        [-81.9, 26.0],
                        [-81.9, 26.1],
                        [-81.95, 26.1],
                        [-81.95, 26.0],
                    ]],
                },
                "properties": {"feature_class": "FAIRWY"},
            }
        ],
    }
    gj_path = tmp_path / "channels.geojson"
    gj_path.write_text(json.dumps(geojson))

    result = rasterize_channels(
        geojson_path=gj_path,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(100, 100),
        out_mask_png=tmp_path / "mask.png",
        out_mask_tif=tmp_path / "mask.tif",
    )

    # Right half should be covered (~50% of pixels).
    assert 0.4 < result.covered_fraction < 0.6
    assert Path(result.mask_path).exists()
    assert Path(result.tif_path).exists()


@pytestmark_cv
def test_rasterize_channels_empty_geojson(tmp_path: Path):
    from readwater.pipeline.channel_mask import rasterize_channels

    geojson = {"type": "FeatureCollection", "features": []}
    gj_path = tmp_path / "empty.geojson"
    gj_path.write_text(json.dumps(geojson))

    result = rasterize_channels(
        geojson_path=gj_path,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(50, 50),
        out_mask_png=tmp_path / "mask.png",
    )

    assert result.covered_fraction == 0.0


@pytestmark_cv
def test_rasterize_channels_multiple_features(tmp_path: Path):
    from readwater.pipeline.channel_mask import rasterize_channels

    # Two small polygons
    geojson = {
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
                "properties": {"feature_class": "FAIRWY"},
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
                "properties": {"feature_class": "DRGARE"},
            },
        ],
    }
    gj_path = tmp_path / "multi.geojson"
    gj_path.write_text(json.dumps(geojson))

    result = rasterize_channels(
        geojson_path=gj_path,
        bbox_4326=(-82.0, 26.0, -81.9, 26.1),
        out_size=(100, 100),
        out_mask_png=tmp_path / "mask.png",
    )

    # Two small polygons should cover a small fraction.
    assert 0 < result.covered_fraction < 0.1
