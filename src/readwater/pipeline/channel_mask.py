"""Rasterize NOAA ENC channel polygons into per-tile binary masks and overlays.

Inputs: a GeoJSON FeatureCollection of channel polygons (produced by
`readwater.api.data_sources.noaa_enc.extract_channels`).

Outputs: a binary raster where 1 = inside a charted navigation channel.
Optionally a georeferenced overlay onto a non-georeferenced base image
(e.g. a NAIP or Google Static PNG) using a known lat/lon bbox.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ChannelRasterResult:
    mask_path: str
    tif_path: str | None
    width: int
    height: int
    covered_fraction: float
    bbox_4326: tuple[float, float, float, float]


def _require_cv_deps():
    try:
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Channel rasterization requires the 'cv' extras. "
            "Install with: pip install -e '.[cv]'"
        ) from e


# ------------------------------------------------------------------
# Rasterize
# ------------------------------------------------------------------


def rasterize_channels(
    geojson_path: str | Path,
    bbox_4326: tuple[float, float, float, float],
    out_size: tuple[int, int],
    out_mask_png: str | Path,
    out_mask_tif: str | Path | None = None,
) -> ChannelRasterResult:
    """Rasterize channel polygons clipped to `bbox_4326` onto a grid of `out_size`.

    Args:
        geojson_path: GeoJSON from extract_channels().
        bbox_4326: (xmin, ymin, xmax, ymax) in WGS84 — the geographic area
            the output raster will cover.
        out_size: (width, height) in pixels for the output raster.
        out_mask_png: path to the output PNG mask (white = channel).
        out_mask_tif: optional georeferenced GeoTIFF output.

    Returns:
        ChannelRasterResult with paths and covered_fraction metrics.
    """
    _require_cv_deps()
    import rasterio
    from rasterio import features
    from rasterio.transform import from_bounds
    from shapely.geometry import shape

    w, h = out_size
    transform = from_bounds(*bbox_4326, w, h)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    # Build shapely geometries
    polys = []
    for feat in geo.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        try:
            shp = shape(g)
        except Exception:  # noqa: BLE001
            continue
        if shp.is_empty:
            continue
        polys.append((shp, 1))

    if not polys:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = features.rasterize(
            ((shp.__geo_interface__, val) for shp, val in polys),
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

    mask_bool = mask > 0

    # Save PNG
    Path(out_mask_png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(out_mask_png)

    tif_out = None
    if out_mask_tif is not None:
        Path(out_mask_tif).parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
        }
        with rasterio.open(out_mask_tif, "w", **profile) as dst:
            dst.write(mask_bool.astype(np.uint8) * 255, indexes=1)
        tif_out = str(out_mask_tif)

    return ChannelRasterResult(
        mask_path=str(out_mask_png),
        tif_path=tif_out,
        width=w,
        height=h,
        covered_fraction=float(mask_bool.mean()),
        bbox_4326=bbox_4326,
    )


# ------------------------------------------------------------------
# Overlay onto a base image (aligned by known bbox)
# ------------------------------------------------------------------


def save_channel_overlay_png(
    rgb_image_path: str | Path,
    mask_bool: np.ndarray,
    rgb_bbox_4326: tuple[float, float, float, float],
    mask_bbox_4326: tuple[float, float, float, float],
    out_path: str | Path,
    channel_rgba: tuple[int, int, int, int] = (255, 0, 0, 120),
    outline_only: bool = False,
) -> str:
    """Overlay a channel mask onto an RGB image, aligned by their bboxes.

    Both the base image and the mask must cover a known lat/lon bbox. The
    mask is resampled into the base image's pixel grid using nearest
    neighbor so the binary character is preserved.

    Args:
        rgb_image_path: PNG/JPEG base image covering `rgb_bbox_4326`.
        mask_bool: boolean array with channel pixels True.
        rgb_bbox_4326: bbox the base image covers.
        mask_bbox_4326: bbox the mask covers. Usually the same as rgb's.
        out_path: output PNG.
        channel_rgba: tint color; red by default since channels are the
            "warning, do not call this a drain" feature.
        outline_only: draw only the boundary of each channel polygon, not
            the full fill. Useful when you want to see the base imagery
            under the channel cells.
    """
    base = Image.open(rgb_image_path).convert("RGBA")
    bw, bh = base.size

    # Resample mask into base's pixel grid using geographic alignment.
    dst_mask = _resample_mask_to_base(
        mask_bool, mask_bbox_4326, rgb_bbox_4326, (bw, bh),
    )

    if outline_only:
        # Simple 4-neighbor boundary detection
        m = dst_mask.astype(np.uint8)
        edge_r = np.roll(m, 1, axis=0) != m
        edge_l = np.roll(m, -1, axis=0) != m
        edge_u = np.roll(m, 1, axis=1) != m
        edge_d = np.roll(m, -1, axis=1) != m
        paint = edge_r | edge_l | edge_u | edge_d
    else:
        paint = dst_mask

    overlay_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    overlay_arr[paint] = channel_rgba
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)


def _resample_mask_to_base(
    mask_bool: np.ndarray,
    mask_bbox: tuple[float, float, float, float],
    base_bbox: tuple[float, float, float, float],
    base_size: tuple[int, int],
) -> np.ndarray:
    """Resample mask_bool (with its own bbox) onto a base pixel grid.

    If the bboxes match exactly, this just resizes via Pillow NEAREST. If
    they differ, the mask is sampled at each base pixel's lat/lon.
    """
    bw, bh = base_size
    mh, mw = mask_bool.shape

    if mask_bbox == base_bbox:
        if (mw, mh) == (bw, bh):
            return mask_bool.astype(bool)
        pil = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
        pil = pil.resize((bw, bh), Image.NEAREST)
        return np.array(pil) > 0

    # General case: sample mask at each base pixel's lat/lon.
    xmin_b, ymin_b, xmax_b, ymax_b = base_bbox
    xmin_m, ymin_m, xmax_m, ymax_m = mask_bbox

    # Build base pixel -> lat/lon grid.
    xs = np.linspace(xmin_b, xmax_b, bw, endpoint=False) + (xmax_b - xmin_b) / (2 * bw)
    ys = np.linspace(ymax_b, ymin_b, bh, endpoint=False) - (ymax_b - ymin_b) / (2 * bh)

    # Map to mask pixel indices (clamped).
    mask_x_idx = ((xs - xmin_m) / (xmax_m - xmin_m) * mw).astype(np.int64)
    mask_y_idx = ((ymax_m - ys) / (ymax_m - ymin_m) * mh).astype(np.int64)

    mask_x_idx = np.clip(mask_x_idx, 0, mw - 1)
    mask_y_idx = np.clip(mask_y_idx, 0, mh - 1)

    ix, iy = np.meshgrid(mask_x_idx, mask_y_idx)
    resampled = mask_bool[iy, ix]

    # Out-of-bbox base pixels should be False (no channel information there).
    out_of_bbox = (
        (xs < xmin_m) | (xs > xmax_m)
    )
    out_of_bbox_y = (ys < ymin_m) | (ys > ymax_m)
    resampled[:, out_of_bbox] = False
    resampled[out_of_bbox_y, :] = False
    return resampled
