"""Placeholder image provider for testing — generates solid-color PNGs."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from readwater.api.providers.base import ImageProvider


def _make_solid_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Generate a minimal solid-color RGB PNG from raw bytes (no Pillow)."""
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_row = bytes([0] + [r, g, b] * width)
    raw_image = raw_row * height
    idat = _chunk(b"IDAT", zlib.compress(raw_image))

    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


class PlaceholderProvider(ImageProvider):
    """Generates solid-color placeholder PNGs for testing without API calls."""

    def __init__(
        self,
        provider_name: str = "placeholder",
        color: tuple[int, int, int] = (100, 149, 237),
        size: int = 8,
    ):
        self._name = provider_name
        self._color = color
        self._size = size

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_zoom(self) -> int:
        return 1

    @property
    def max_zoom(self) -> int:
        return 20

    async def fetch(
        self,
        center: tuple[float, float],
        zoom: int,
        output_path: str,
        image_size: int = 640,
    ) -> str:
        data = _make_solid_png(self._size, self._size, *self._color)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)
