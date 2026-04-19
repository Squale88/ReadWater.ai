"""4-band NAIP fetcher via Microsoft Planetary Computer STAC.

NAIP 4-band imagery (Red, Green, Blue, Near-Infrared) is the input for
NDWI water masks, NDVI vegetation masks, and other bottom/land
discriminations. The USGS ImageServer exports only RGB composites; the raw
4-band COGs are hosted on Planetary Computer as a STAC collection.

This module downloads a windowed 4-band GeoTIFF for a given bbox (or
lat/lon + zoom/size) and saves it to disk. Dependencies are optional —
rasterio + pystac-client + planetary-computer are imported lazily so the
rest of the codebase runs without them.

Coverage: continental US only. Typical latest acquisition is 2022-2023 for
most Florida counties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

EARTH_CIRCUMFERENCE_MILES = 24901.0
MILES_PER_DEG_LAT = 69.0

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "naip"


@dataclass
class FourBandResult:
    """Metadata returned after fetching a 4-band NAIP tile."""

    path: str
    acquired_year: int
    item_id: str
    bands: int
    height: int
    width: int
    bbox_4326: tuple[float, float, float, float]
    nir_band_index: int   # 1-based band index of NIR in the output file


def bbox_from_center(
    center: tuple[float, float], zoom: int, image_size: int = 640,
) -> tuple[float, float, float, float]:
    """Convenience: compute a lat/lon bbox for a Google-Static-equivalent tile."""
    lat, lon = center
    tiles = image_size / 256
    span_miles = (
        tiles * EARTH_CIRCUMFERENCE_MILES * math.cos(math.radians(lat)) / (2**zoom)
    )
    half_lat = (span_miles / 2) / MILES_PER_DEG_LAT
    cos_lat = math.cos(math.radians(lat))
    half_lon = (span_miles / 2) / (MILES_PER_DEG_LAT * cos_lat) if cos_lat > 1e-6 else half_lat
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def _require_cv_deps():
    try:
        import numpy as np  # noqa: F401
        import pystac_client  # noqa: F401
        import planetary_computer  # noqa: F401
        import rasterio  # noqa: F401
        from rasterio.mask import mask  # noqa: F401
        from rasterio.warp import transform_bounds  # noqa: F401
        from rasterio.windows import from_bounds  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "4-band NAIP requires the 'cv' extras. "
            "Install with: pip install -e '.[cv]'"
        ) from e


def fetch_naip_4band(
    bbox_4326: tuple[float, float, float, float],
    out_path: str | Path,
    year: int | None = None,
    prefer_latest: bool = True,
) -> FourBandResult:
    """Download a windowed 4-band NAIP GeoTIFF covering `bbox_4326`.

    NAIP is acquired in ~3.75' × 3.75' quarter-quads, so a single item
    usually covers only part of an arbitrary bbox. This function searches
    Planetary Computer for ALL items overlapping the bbox, keeps only the
    most-recent-year items (so we don't mix acquisition years), and
    mosaics them into a single output covering the full bbox.

    Args:
        bbox_4326: (xmin, ymin, xmax, ymax) in WGS84 lat/lon.
        out_path: output path for the GeoTIFF.
        year: optional filter for acquisition year. None uses most-recent.
        prefer_latest: if True, use the most recent year's items only.

    Returns:
        FourBandResult with metadata including which items were merged.
    """
    _require_cv_deps()
    import pystac_client
    import planetary_computer
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox_4326,
        limit=50,
    )
    items = list(search.items())
    if year is not None:
        items = [i for i in items if i.datetime and i.datetime.year == year]
    if not items:
        raise RuntimeError(
            f"no NAIP items for bbox {bbox_4326}"
            + (f" in year {year}" if year else "")
        )

    # Restrict to a single acquisition year so the mosaic is self-consistent.
    if prefer_latest and year is None:
        items.sort(key=lambda i: i.datetime or 0, reverse=True)
        chosen_year = items[0].datetime.year if items[0].datetime else None
        items = [i for i in items if i.datetime and i.datetime.year == chosen_year]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open every item's image asset; they share a CRS within a state/year.
    datasets = []
    item_ids = []
    for item in items:
        try:
            ds = rasterio.open(item.assets["image"].href)
            datasets.append(ds)
            item_ids.append(item.id)
        except Exception:  # noqa: BLE001
            # Skip unreachable items (e.g. transient signed URL expiry)
            continue
    if not datasets:
        raise RuntimeError(f"could not open any NAIP items for bbox {bbox_4326}")

    # Ensure all items share a CRS; reject otherwise so the merge is clean.
    crs_set = {ds.crs for ds in datasets}
    if len(crs_set) > 1:
        # Keep the subset matching the first (most-recent) item's CRS.
        target_crs = datasets[0].crs
        filtered = [(ds, iid) for ds, iid in zip(datasets, item_ids) if ds.crs == target_crs]
        # Close the ones we're dropping.
        for ds, iid in zip(datasets, item_ids):
            if ds.crs != target_crs:
                ds.close()
        if not filtered:
            raise RuntimeError(
                f"no consistent-CRS NAIP items available for bbox {bbox_4326}",
            )
        datasets = [p[0] for p in filtered]
        item_ids = [p[1] for p in filtered]

    dst_crs = datasets[0].crs
    merge_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_4326)

    # Mosaic: rasterio.merge selects pixel values from the first dataset
    # available at each output pixel.
    try:
        data, out_transform = merge(datasets, bounds=merge_bounds)
    finally:
        for ds in datasets:
            ds.close()

    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": dst_crs,
        "transform": out_transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    latest_year = max(
        (i.datetime.year for i in items if i.datetime), default=0,
    )

    # NAIP convention: band 1 = Red, 2 = Green, 3 = Blue, 4 = NIR.
    result = FourBandResult(
        path=str(out_path),
        acquired_year=latest_year,
        item_id=",".join(item_ids),
        bands=int(data.shape[0]),
        height=int(data.shape[1]),
        width=int(data.shape[2]),
        bbox_4326=bbox_4326,
        nir_band_index=4,
    )
    return result
