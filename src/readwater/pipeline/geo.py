"""Web-Mercator pixel <-> lat/lon helpers for Google Static Maps imagery.

Google Static Maps with `scale=2` and a requested `size=640x640` returns a
1280x1280 pixel image that covers the same ground as a `scale=1` image of the
same size parameter. In Mercator terms that means the pixel resolution is
twice as fine as the nominal zoom, i.e. "effective pixel zoom" = zoom + 1.

All formulas here are the standard Web-Mercator approximations:
    meters_per_pixel(z, lat) = 156543.03392 * cos(lat_rad) / 2**(z+1)
    deg_lat_per_pixel(z, lat) = meters_per_pixel / 111320
    deg_lon_per_pixel(z, lat) = deg_lat_per_pixel / cos(lat_rad)

The +1 in the exponent accounts for scale=2. Pure Mercator distortion is fine
at continental-US latitudes; above ~70deg the flat-earth approximation breaks
down.

All functions are pure; no I/O.
"""

from __future__ import annotations

import math

# Earth equatorial circumference in meters / 256 px tile at zoom 0.
METERS_PER_TILE_AT_EQUATOR_ZOOM_0 = 156543.03392
METERS_PER_DEG_LAT = 111320.0
GOOGLE_STATIC_SCALE = 2  # scale=2 everywhere in this codebase


def _effective_zoom(zoom: int, scale: int = GOOGLE_STATIC_SCALE) -> int:
    """Pixel-density zoom: scale=2 doubles resolution, so add log2(scale)."""
    return zoom + int(math.log2(scale))


def meters_per_pixel(zoom: int, lat: float, scale: int = GOOGLE_STATIC_SCALE) -> float:
    """Ground distance per pixel at a given zoom and latitude, for scale=scale imagery."""
    eff = _effective_zoom(zoom, scale)
    return METERS_PER_TILE_AT_EQUATOR_ZOOM_0 * math.cos(math.radians(lat)) / (2**eff)


def deg_lat_per_pixel(zoom: int, lat: float, scale: int = GOOGLE_STATIC_SCALE) -> float:
    """Latitude degrees per pixel (vertical axis) at the given zoom/lat."""
    return meters_per_pixel(zoom, lat, scale) / METERS_PER_DEG_LAT


def deg_lon_per_pixel(zoom: int, lat: float, scale: int = GOOGLE_STATIC_SCALE) -> float:
    """Longitude degrees per pixel (horizontal axis) at the given zoom/lat."""
    cos_lat = math.cos(math.radians(lat))
    if cos_lat < 1e-12:
        # At the poles longitude is meaningless. Clamp to a huge but finite value.
        return float("inf")
    return deg_lat_per_pixel(zoom, lat, scale) / cos_lat


def pixel_to_latlon(
    px: float,
    py: float,
    img_size_px: int,
    center_lat: float,
    center_lon: float,
    zoom: int,
    scale: int = GOOGLE_STATIC_SCALE,
) -> tuple[float, float]:
    """Convert a pixel coordinate in an image to (lat, lon).

    The image is assumed centered on (center_lat, center_lon). Pixel origin
    (0, 0) is the top-left; py increases southward (decreasing latitude).

    img_size_px is the full image size in pixels (e.g. 1280 for a Google
    Static `size=640` + `scale=2` fetch).
    """
    half = img_size_px / 2.0
    dx = px - half
    dy = py - half

    lat = center_lat - dy * deg_lat_per_pixel(zoom, center_lat, scale)
    lon = center_lon + dx * deg_lon_per_pixel(zoom, center_lat, scale)
    return (lat, lon)


def latlon_to_pixel(
    lat: float,
    lon: float,
    img_size_px: int,
    center_lat: float,
    center_lon: float,
    zoom: int,
    scale: int = GOOGLE_STATIC_SCALE,
) -> tuple[float, float]:
    """Convert a (lat, lon) back to a pixel coordinate in an image centered on (center_lat, center_lon).

    Inverse of pixel_to_latlon. Both degree-per-pixel factors are evaluated at
    the image center latitude; this is the standard flat-Mercator-window
    approximation used by Google Static Maps for small extents.
    """
    half = img_size_px / 2.0
    dlat = lat - center_lat
    dlon = lon - center_lon

    dy = -dlat / deg_lat_per_pixel(zoom, center_lat, scale)
    dx = dlon / deg_lon_per_pixel(zoom, center_lat, scale)
    return (half + dx, half + dy)


def polygon_px_to_latlon(
    polygon_px: list[tuple[float, float]],
    img_size_px: int,
    center_lat: float,
    center_lon: float,
    zoom: int,
    scale: int = GOOGLE_STATIC_SCALE,
) -> list[tuple[float, float]]:
    """Convert a pixel polygon to a lat/lon polygon, preserving vertex order."""
    return [
        pixel_to_latlon(x, y, img_size_px, center_lat, center_lon, zoom, scale)
        for (x, y) in polygon_px
    ]


def clip_polygon_to_rect(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Clip a polygon to the rectangle [0, width] x [0, height] using Sutherland-Hodgman.

    Returns integer pixel coordinates. If the result has fewer than 3 vertices
    the polygon is considered collapsed and an empty list is returned.
    """
    if not polygon:
        return []

    def _inside(p: tuple[float, float], edge: str) -> bool:
        x, y = p
        if edge == "left":
            return x >= 0
        if edge == "right":
            return x <= width
        if edge == "top":
            return y >= 0
        if edge == "bottom":
            return y <= height
        raise ValueError(edge)

    def _intersect(
        a: tuple[float, float], b: tuple[float, float], edge: str,
    ) -> tuple[float, float]:
        ax, ay = a
        bx, by = b
        if edge in ("left", "right"):
            x_edge = 0 if edge == "left" else width
            if bx == ax:
                return (x_edge, ay)
            t = (x_edge - ax) / (bx - ax)
            return (x_edge, ay + t * (by - ay))
        y_edge = 0 if edge == "top" else height
        if by == ay:
            return (ax, y_edge)
        t = (y_edge - ay) / (by - ay)
        return (ax + t * (bx - ax), y_edge)

    output = list(polygon)
    for edge in ("left", "right", "top", "bottom"):
        if not output:
            break
        inp = output
        output = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if _inside(e, edge):
                if not _inside(s, edge):
                    output.append(_intersect(s, e, edge))
                output.append(e)
            elif _inside(s, edge):
                output.append(_intersect(s, e, edge))
            s = e

    if len(output) < 3:
        return []
    return [(int(round(x)), int(round(y))) for (x, y) in output]
