"""NAIP imagery provider — free USDA aerial imagery via USGS ArcGIS ImageServer.

NAIP (National Agriculture Imagery Program) covers the continental US at
60 cm resolution, updated every 2-3 years. It is public domain, requires no
API key, and substantially beats Google Static at discriminating water,
vegetation, and exposed bottom due to higher native resolution.

This provider returns the default RGB (natural color) composite, matching
GoogleStaticProvider's drop-in interface. For 4-band (RGB+NIR) access used
by the water-mask pipeline, see `api.data_sources.naip_4band`.
"""

from __future__ import annotations

import math
from pathlib import Path

import httpx

from readwater.api.providers.base import ImageProvider

# USGS TNM NAIP ImageServer. Public, no auth.
NAIP_EXPORT_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPImagery/ImageServer/exportImage"
)

# Match GoogleStaticProvider's geodesy constants so a 640-size request
# covers the same ground for both providers.
EARTH_CIRCUMFERENCE_MILES = 24901.0
MILES_PER_DEG_LAT = 69.0


def _ground_span_miles(zoom: int, lat: float, image_size: int) -> float:
    """Miles covered by an image_size-pixel side at this zoom/lat (256 px tile base)."""
    tiles = image_size / 256
    return tiles * EARTH_CIRCUMFERENCE_MILES * math.cos(math.radians(lat)) / (2**zoom)


def _bbox_from_center(
    center: tuple[float, float], zoom: int, image_size: int,
) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in EPSG:4326 for a Google-Static-equivalent tile."""
    lat, lon = center
    span_miles = _ground_span_miles(zoom, lat, image_size)
    half_lat = (span_miles / 2) / MILES_PER_DEG_LAT
    cos_lat = math.cos(math.radians(lat))
    half_lon = (span_miles / 2) / (MILES_PER_DEG_LAT * cos_lat) if cos_lat > 1e-6 else half_lat
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


class NAIPProvider(ImageProvider):
    """Fetches NAIP aerial imagery (natural color RGB) from USGS ImageServer.

    Scale convention matches GoogleStaticProvider: an `image_size=640` request
    returns a 1280x1280 PNG (same ground as a scale=2 Google Static tile).
    """

    def __init__(
        self,
        timeout_s: float = 60.0,
        output_format: str = "png",
        pixel_multiplier: int = 2,
    ):
        self._timeout_s = timeout_s
        if output_format not in ("png", "jpg", "jpeg", "tiff"):
            raise ValueError(f"unsupported format {output_format}")
        self._format = output_format
        self._pixel_multiplier = pixel_multiplier

    @property
    def name(self) -> str:
        return "naip"

    @property
    def min_zoom(self) -> int:
        # NAIP's 60 cm resolution supports usable rendering from zoom 14 up.
        # At zoom 10-13 the service still responds but content is smoothed; we
        # keep Google Static as the overview-zoom provider.
        return 14

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
        bbox = _bbox_from_center(center, zoom, image_size)
        pixels = image_size * self._pixel_multiplier
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "bboxSR": "4326",
            "size": f"{pixels},{pixels}",
            "imageSR": "3857",
            "format": self._format,
            "f": "image",
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                NAIP_EXPORT_URL, params=params, timeout=self._timeout_s,
            )
            response.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(response.content)
        return output_path
