"""Tests for the image provider abstraction layer."""

import os
from pathlib import Path

import pytest

from readwater.api.providers.base import ImageProvider
from readwater.api.providers.google_static import GoogleStaticProvider
from readwater.api.providers.placeholder import PlaceholderProvider, _make_solid_png
from readwater.api.providers.registry import ImageProviderRegistry

MARCO = (25.94, -81.73)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --- PlaceholderProvider ---


def test_placeholder_default_name():
    assert PlaceholderProvider().name == "placeholder"


def test_placeholder_custom_name():
    p = PlaceholderProvider(provider_name="naip_test")
    assert p.name == "naip_test"


def test_placeholder_zoom_range():
    p = PlaceholderProvider()
    assert p.min_zoom == 1
    assert p.max_zoom == 20


def test_placeholder_supports_zoom():
    p = PlaceholderProvider()
    assert p.supports_zoom(10) is True
    assert p.supports_zoom(1) is True
    assert p.supports_zoom(20) is True
    assert p.supports_zoom(0) is False
    assert p.supports_zoom(21) is False


async def test_placeholder_fetch_creates_file(tmp_path):
    p = PlaceholderProvider()
    out = str(tmp_path / "test.png")
    result = await p.fetch(MARCO, zoom=12, output_path=out)
    assert Path(result).exists()


async def test_placeholder_fetch_writes_valid_png(tmp_path):
    p = PlaceholderProvider()
    out = str(tmp_path / "test.png")
    await p.fetch(MARCO, zoom=12, output_path=out)
    data = Path(out).read_bytes()
    assert data[:8] == PNG_MAGIC


async def test_placeholder_fetch_returns_path(tmp_path):
    p = PlaceholderProvider()
    out = str(tmp_path / "test.png")
    result = await p.fetch(MARCO, zoom=12, output_path=out)
    assert result == out


async def test_placeholder_creates_parent_dirs(tmp_path):
    p = PlaceholderProvider()
    out = str(tmp_path / "nested" / "deep" / "test.png")
    await p.fetch(MARCO, zoom=12, output_path=out)
    assert Path(out).exists()


# --- Raw PNG generation ---


def test_make_solid_png_valid():
    data = _make_solid_png(8, 8, 100, 149, 237)
    assert data[:8] == PNG_MAGIC
    assert len(data) > 50


# --- GoogleStaticProvider ---


def test_google_static_name():
    assert GoogleStaticProvider().name == "google_static"


def test_google_static_zoom_range():
    p = GoogleStaticProvider()
    assert p.min_zoom == 1
    assert p.max_zoom == 20


def test_google_static_supports_zoom():
    p = GoogleStaticProvider()
    assert p.supports_zoom(15) is True
    assert p.supports_zoom(0) is False


def test_google_static_raises_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_MAPS_API_KEY"):
        GoogleStaticProvider._get_api_key()


# --- ImageProviderRegistry ---


def test_registry_register_and_get():
    reg = ImageProviderRegistry()
    p = PlaceholderProvider()
    reg.register(p, ["overview"])
    assert reg.get_providers("overview") == [p]


def test_registry_multiple_providers_same_role():
    reg = ImageProviderRegistry()
    p1 = PlaceholderProvider(provider_name="a")
    p2 = PlaceholderProvider(provider_name="b")
    reg.register(p1, ["structure"])
    reg.register(p2, ["structure"])
    providers = reg.get_providers("structure")
    assert len(providers) == 2
    assert providers[0].name == "a"
    assert providers[1].name == "b"


def test_registry_get_default_returns_first():
    reg = ImageProviderRegistry()
    p1 = PlaceholderProvider(provider_name="first")
    p2 = PlaceholderProvider(provider_name="second")
    reg.register(p1, ["overview"])
    reg.register(p2, ["overview"])
    assert reg.get_default_provider("overview").name == "first"


def test_registry_missing_role_raises():
    reg = ImageProviderRegistry()
    with pytest.raises(ValueError, match="No providers registered"):
        reg.get_providers("nonexistent")


def test_registry_get_default_missing_raises():
    reg = ImageProviderRegistry()
    with pytest.raises(ValueError, match="No providers registered"):
        reg.get_default_provider("nonexistent")


def test_registry_provider_multiple_roles():
    reg = ImageProviderRegistry()
    p = PlaceholderProvider()
    reg.register(p, ["overview", "structure"])
    assert reg.get_providers("overview") == [p]
    assert reg.get_providers("structure") == [p]


# --- Integration test (real API) ---


@pytest.mark.integration
async def test_google_static_fetch_real_image(tmp_path):
    """Fetch one real satellite image of Marco Island at zoom 11.

    Invoke with: pytest -m integration
    """
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        pytest.skip("GOOGLE_MAPS_API_KEY not set")

    p = GoogleStaticProvider()
    out = str(tmp_path / "marco.png")
    result = await p.fetch(MARCO, zoom=11, output_path=out)

    assert result == out
    data = Path(out).read_bytes()
    assert len(data) > 1000
    assert data[:8] == PNG_MAGIC
