"""Tests for grid overlay image processing."""

from pathlib import Path

from PIL import Image

from readwater.api.providers.placeholder import PlaceholderProvider
from readwater.pipeline.image_processing import draw_grid_overlay

MARCO = (25.94, -81.73)


async def _make_test_image(tmp_path: Path, name: str = "test.png") -> str:
    """Generate a test image via PlaceholderProvider."""
    p = PlaceholderProvider(size=640)
    out = str(tmp_path / name)
    await p.fetch(MARCO, zoom=10, output_path=out)
    return out


async def test_grid_overlay_creates_file(tmp_path):
    src = await _make_test_image(tmp_path)
    result = draw_grid_overlay(src)
    assert Path(result).exists()


async def test_grid_overlay_default_suffix(tmp_path):
    src = await _make_test_image(tmp_path, "z1_r0c1.png")
    result = draw_grid_overlay(src)
    assert Path(result).name == "z1_r0c1_grid.png"


async def test_grid_overlay_original_unchanged(tmp_path):
    src = await _make_test_image(tmp_path)
    original_bytes = Path(src).read_bytes()
    draw_grid_overlay(src)
    assert Path(src).read_bytes() == original_bytes


async def test_grid_overlay_valid_png(tmp_path):
    src = await _make_test_image(tmp_path)
    result = draw_grid_overlay(src)
    data = Path(result).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


async def test_grid_overlay_same_dimensions(tmp_path):
    src = await _make_test_image(tmp_path)
    result = draw_grid_overlay(src)
    orig = Image.open(src)
    grid = Image.open(result)
    assert orig.size == grid.size


async def test_grid_overlay_custom_output_path(tmp_path):
    src = await _make_test_image(tmp_path)
    custom = str(tmp_path / "custom" / "grid.png")
    result = draw_grid_overlay(src, output_path=custom)
    assert result == custom
    assert Path(custom).exists()


async def test_grid_overlay_different_from_original(tmp_path):
    """Grid overlay should modify pixel data (lines + numbers drawn)."""
    src = await _make_test_image(tmp_path)
    result = draw_grid_overlay(src)
    assert Path(src).read_bytes() != Path(result).read_bytes()
