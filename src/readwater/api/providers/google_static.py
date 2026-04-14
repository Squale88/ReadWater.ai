"""Google Maps Static API satellite imagery provider."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from readwater.api.providers.base import ImageProvider

GOOGLE_MAPS_STATIC_URL = "https://maps.googleapis.com/maps/api/staticmap"


class GoogleStaticProvider(ImageProvider):
    """Fetches satellite imagery from Google Maps Static API."""

    @property
    def name(self) -> str:
        return "google_static"

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
        lat, lon = center
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{image_size}x{image_size}",
            "scale": 2,
            "maptype": "satellite",
            "key": self._get_api_key(),
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(GOOGLE_MAPS_STATIC_URL, params=params)
            resp.raise_for_status()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return str(path)

    @staticmethod
    def _get_api_key() -> str:
        key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable is not set")
        return key
