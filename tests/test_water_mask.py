"""Tests for the NDWI water-mask module (pure-numpy portions).

The rasterio-backed load/save helpers are integration-tested separately
and gated by availability of the 'cv' extras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from readwater.pipeline.water_mask import (
    compute_ndwi,
    save_mask_overlay_png,
    save_mask_png,
    threshold_water,
)


# --- NDWI correctness ---


def test_ndwi_pure_water_strongly_positive():
    # Water: high green reflectance, near-zero NIR.
    green = np.full((4, 4), 0.4, dtype=np.float32)
    nir = np.full((4, 4), 0.05, dtype=np.float32)
    ndwi = compute_ndwi(green, nir)
    # (0.4 - 0.05) / (0.4 + 0.05) ~= 0.777
    assert np.all(ndwi > 0.7)


def test_ndwi_pure_vegetation_strongly_negative():
    # Vegetation: moderate green, high NIR.
    green = np.full((4, 4), 0.25, dtype=np.float32)
    nir = np.full((4, 4), 0.55, dtype=np.float32)
    ndwi = compute_ndwi(green, nir)
    # (0.25 - 0.55) / (0.25 + 0.55) = -0.375
    assert np.all(ndwi < -0.3)


def test_ndwi_zero_divzero_safe():
    green = np.zeros((3, 3), dtype=np.float32)
    nir = np.zeros((3, 3), dtype=np.float32)
    ndwi = compute_ndwi(green, nir)
    # All zeros in both bands -> denom is zero; output should be zero, not inf/nan
    assert np.all(np.isfinite(ndwi))
    assert np.all(ndwi == 0.0)


def test_ndwi_accepts_uint8_input():
    """Callers may pass raw uint8 pixels; compute_ndwi should not overflow."""
    green = np.full((4, 4), 200, dtype=np.uint8)
    nir = np.full((4, 4), 50, dtype=np.uint8)
    ndwi = compute_ndwi(green, nir)
    assert np.all(ndwi > 0.5)


# --- threshold_water ---


def test_threshold_water_picks_above_value():
    ndwi = np.array([[-0.3, 0.1, 0.5], [0.0, 0.2, -0.1]], dtype=np.float32)
    mask = threshold_water(ndwi, threshold=0.0)
    expected = np.array([[False, True, True], [False, True, False]], dtype=bool)
    assert (mask == expected).all()


def test_threshold_water_conservative_threshold():
    ndwi = np.array([[0.05, 0.12, 0.20]], dtype=np.float32)
    mask = threshold_water(ndwi, threshold=0.10)
    assert mask.tolist() == [[False, True, True]]


def test_threshold_water_morphology_kills_speckle():
    # Single isolated pixel surrounded by non-water; morphology should remove.
    ndwi = np.full((10, 10), -0.5, dtype=np.float32)
    ndwi[5, 5] = 0.9
    mask_no_morph = threshold_water(ndwi, threshold=0.0, min_run_pixels=0)
    assert mask_no_morph[5, 5]
    mask_morph = threshold_water(ndwi, threshold=0.0, min_run_pixels=1)
    assert not mask_morph[5, 5]


def test_threshold_water_morphology_preserves_large_region():
    ndwi = np.full((10, 10), -0.5, dtype=np.float32)
    ndwi[2:8, 2:8] = 0.9
    mask = threshold_water(ndwi, threshold=0.0, min_run_pixels=2)
    # Erosion then dilation on a 6x6 block preserves most of the interior.
    assert mask[4, 4]
    assert mask[5, 5]


# --- PNG helpers ---


def test_save_mask_png_roundtrip(tmp_path: Path):
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    out = tmp_path / "mask.png"
    save_mask_png(mask, out)
    assert out.exists()

    img = Image.open(out).convert("L")
    arr = np.array(img)
    # Water pixels should be 255, non-water 0.
    assert arr.shape == mask.shape
    assert (arr[mask] == 255).all()
    assert (arr[~mask] == 0).all()


def test_save_mask_overlay_matches_base_size(tmp_path: Path):
    # Base image 40x40 RGB.
    base = Image.new("RGB", (40, 40), (100, 150, 200))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    # Mask at different resolution (8x8) — overlay helper should resize.
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    out = tmp_path / "overlay.png"
    save_mask_overlay_png(base_path, mask, out)

    assert out.exists()
    overlay = Image.open(out)
    assert overlay.size == (40, 40)


def test_save_mask_overlay_tints_only_water_pixels(tmp_path: Path):
    base = Image.new("RGB", (20, 20), (100, 100, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :] = True  # Top half is water.

    out = tmp_path / "overlay.png"
    save_mask_overlay_png(base_path, mask, out, water_rgba=(0, 200, 255, 200))

    arr = np.array(Image.open(out).convert("RGB"))
    # Top half should be tinted blue (shifted from 100,100,100).
    top_mean = arr[:10].mean(axis=(0, 1))
    bot_mean = arr[10:].mean(axis=(0, 1))
    # Blue channel should be higher on top due to the blue tint.
    assert top_mean[2] > bot_mean[2] + 20
    # Bottom untouched (roughly original grey).
    assert bot_mean[0] == pytest.approx(100, abs=2)
