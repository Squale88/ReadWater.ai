"""Tests for the NAIP imagery provider (USGS ImageServer RGB path)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from readwater.api.providers.naip import (
    NAIPProvider,
    _bbox_from_center,
    _ground_span_miles,
)


def _png_bytes() -> bytes:
    """A minimal valid 1x1 PNG for mocking HTTP responses."""
    # Hard-coded minimal PNG; decoded later by Pillow if needed.
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452"
        "00000001000000010802000000907753de"
        "0000000a49444154789c63000100000005"
        "000105d8c57b0000000049454e44ae426082"
    )


# --- Geodesy helpers ---


def test_ground_span_scales_with_zoom():
    """Each zoom-level increase halves the ground coverage."""
    a = _ground_span_miles(16, 26.0, 640)
    b = _ground_span_miles(17, 26.0, 640)
    assert a == pytest.approx(b * 2, rel=1e-6)


def test_ground_span_shrinks_with_latitude():
    """Higher latitudes have shorter longitudinal spans at the same zoom."""
    eq = _ground_span_miles(16, 0.0, 640)
    mid = _ground_span_miles(16, 45.0, 640)
    high = _ground_span_miles(16, 60.0, 640)
    assert eq > mid > high


def test_bbox_from_center_centers_on_point():
    lat, lon = 26.011172, -81.753546
    xmin, ymin, xmax, ymax = _bbox_from_center((lat, lon), zoom=16, image_size=640)
    assert pytest.approx((xmin + xmax) / 2, abs=1e-9) == lon
    assert pytest.approx((ymin + ymax) / 2, abs=1e-9) == lat
    assert xmax > xmin
    assert ymax > ymin


def test_bbox_width_reasonable_at_zoom_16():
    """~0.85 mi ground coverage at zoom 16 = ~0.0124 deg lat for a 640 size."""
    xmin, ymin, xmax, ymax = _bbox_from_center(
        (26.0, -81.75), zoom=16, image_size=640,
    )
    # 0.85 mi / 69 mi-per-deg ~= 0.0123 deg lat
    assert (ymax - ymin) == pytest.approx(0.0123, abs=0.001)


# --- Provider interface ---


def test_provider_name_and_zoom_range():
    p = NAIPProvider()
    assert p.name == "naip"
    assert p.min_zoom == 14
    assert p.max_zoom == 20
    assert p.supports_zoom(16)
    assert not p.supports_zoom(10)
    assert not p.supports_zoom(21)


def test_invalid_format_raises():
    with pytest.raises(ValueError):
        NAIPProvider(output_format="bmp")


# --- fetch() — mocked HTTP ---


async def test_fetch_writes_png_and_passes_bbox(tmp_path: Path):
    provider = NAIPProvider()
    out = tmp_path / "naip.png"

    mock_resp = MagicMock()
    mock_resp.content = _png_bytes()
    mock_resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("readwater.api.providers.naip.httpx.AsyncClient", return_value=mock_client):
        result = await provider.fetch(
            center=(26.011172, -81.753546),
            zoom=16,
            output_path=str(out),
            image_size=640,
        )

    assert result == str(out)
    assert out.exists()
    assert out.read_bytes() == _png_bytes()

    # Validate request params
    mock_client.get.assert_called_once()
    call_args = mock_client.get.call_args
    params = call_args.kwargs.get("params") or call_args[1]["params"]
    # bbox is 4 comma-separated floats in 4326
    assert params["bboxSR"] == "4326"
    assert params["imageSR"] == "3857"
    assert params["format"] == "png"
    assert params["f"] == "image"
    # size is image_size * pixel_multiplier (default 2)
    assert params["size"] == "1280,1280"
    # bbox values are parseable
    bbox_parts = [float(v) for v in params["bbox"].split(",")]
    assert len(bbox_parts) == 4
    # Center of bbox equals requested lat/lon
    xmin, ymin, xmax, ymax = bbox_parts
    assert (xmin + xmax) / 2 == pytest.approx(-81.753546, abs=1e-6)
    assert (ymin + ymax) / 2 == pytest.approx(26.011172, abs=1e-6)


async def test_fetch_respects_image_size(tmp_path: Path):
    provider = NAIPProvider(pixel_multiplier=1)
    out = tmp_path / "naip.png"

    mock_resp = MagicMock()
    mock_resp.content = _png_bytes()
    mock_resp.raise_for_status = MagicMock(return_value=None)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("readwater.api.providers.naip.httpx.AsyncClient", return_value=mock_client):
        await provider.fetch(
            center=(26.0, -81.0), zoom=16, output_path=str(out), image_size=320,
        )

    params = mock_client.get.call_args.kwargs.get("params") or mock_client.get.call_args[1]["params"]
    # pixel_multiplier=1, image_size=320 -> 320x320
    assert params["size"] == "320,320"


async def test_fetch_creates_parent_dir(tmp_path: Path):
    provider = NAIPProvider()
    out = tmp_path / "nested" / "dirs" / "naip.png"

    mock_resp = MagicMock()
    mock_resp.content = _png_bytes()
    mock_resp.raise_for_status = MagicMock(return_value=None)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("readwater.api.providers.naip.httpx.AsyncClient", return_value=mock_client):
        await provider.fetch(
            center=(26.0, -81.0), zoom=16, output_path=str(out),
        )

    assert out.exists()
