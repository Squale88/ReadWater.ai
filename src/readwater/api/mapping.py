"""Satellite imagery fetching via Google Maps Static API."""

from __future__ import annotations

import os

import httpx

from readwater.models.cell import BoundingBox

GOOGLE_MAPS_STATIC_URL = "https://maps.googleapis.com/maps/api/staticmap"


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable is not set")
    return key


def _estimate_zoom_level(size_miles: float, image_size_px: int = 640) -> int:
    """Estimate the Google Maps zoom level needed to fit a given area.

    Rough heuristic: at zoom 10, ~30 miles fits; each zoom level halves the area.
    """
    import math

    if size_miles <= 0:
        return 20
    zoom = 10 + math.log2(30 / size_miles)
    return max(1, min(20, int(round(zoom))))


async def fetch_satellite_image(
    bbox: BoundingBox,
    size_px: int = 640,
) -> bytes:
    """Fetch a satellite image for the given bounding box.

    Args:
        bbox: Geographic bounding box to capture.
        size_px: Image dimension in pixels (max 640 for free tier).

    Returns:
        Raw image bytes (PNG).
    """
    center_lat, center_lon = bbox.center
    # Estimate the cell's size in miles from the bbox
    lat_span = bbox.north - bbox.south
    size_miles = lat_span / MILES_TO_LAT_DEG
    zoom = _estimate_zoom_level(size_miles, size_px)

    params = {
        "center": f"{center_lat},{center_lon}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",
        "maptype": "satellite",
        "key": _get_api_key(),
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(GOOGLE_MAPS_STATIC_URL, params=params)
        resp.raise_for_status()
        return resp.content


# Re-export for use in mapping module calculations
MILES_TO_LAT_DEG = 1 / 69.0
