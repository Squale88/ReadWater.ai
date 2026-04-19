"""Generic polygon-based raster mask pipeline.

Used by every ground-truth overlay in the project:
  - NOAA ENC navigation channels
  - FWC oyster reef surveys
  - FWC seagrass surveys
  - anything future that starts as a GeoJSON FeatureCollection of polygons

Given a GeoJSON, a geographic bbox, and a target pixel grid, produces:
  - a per-tile binary raster aligned to the tile grid (PNG)
  - optionally the same binary raster as a georeferenced GeoTIFF
  - an overlay onto a non-georeferenced RGB base image using a known bbox

All bboxes are (xmin, ymin, xmax, ymax) in EPSG:4326 unless noted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PolygonRasterResult:
    mask_path: str
    tif_path: str | None
    width: int
    height: int
    covered_fraction: float
    bbox_4326: tuple[float, float, float, float]


def _require_cv_deps() -> None:
    try:
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Polygon rasterization requires the 'cv' extras. "
            "Install with: pip install -e '.[cv]'"
        ) from e


# ------------------------------------------------------------------
# Rasterize
# ------------------------------------------------------------------


def rasterize_polygons(
    geojson_path: str | Path,
    bbox_4326: tuple[float, float, float, float],
    out_size: tuple[int, int],
    out_mask_png: str | Path,
    out_mask_tif: str | Path | None = None,
) -> PolygonRasterResult:
    """Rasterize polygons in `geojson_path` clipped to `bbox_4326` onto `out_size`.

    Args:
        geojson_path: GeoJSON FeatureCollection of polygons (any source).
        bbox_4326: (xmin, ymin, xmax, ymax) the output raster will cover.
        out_size: (width, height) in pixels.
        out_mask_png: path for the 8-bit PNG mask (white = feature).
        out_mask_tif: optional georeferenced GeoTIFF output.
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

    return PolygonRasterResult(
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


def save_polygon_overlay_png(
    rgb_image_path: str | Path,
    mask_bool: np.ndarray,
    rgb_bbox_4326: tuple[float, float, float, float],
    mask_bbox_4326: tuple[float, float, float, float],
    out_path: str | Path,
    rgba: tuple[int, int, int, int],
    outline_only: bool = False,
) -> str:
    """Overlay a boolean mask onto an RGB image, aligned by geographic bbox.

    Args:
        rgb_image_path: PNG/JPEG base (assumed to cover `rgb_bbox_4326`).
        mask_bool: boolean array with feature pixels True.
        rgb_bbox_4326: bbox the base image covers.
        mask_bbox_4326: bbox the mask covers. Usually the same as rgb's.
        out_path: output PNG.
        rgba: tint color (R, G, B, A). Callers supply this to distinguish
            overlay types visually (e.g. channels red, oysters purple).
        outline_only: draw only the boundary of each polygon. Useful when
            you want to see the base imagery under the polygons.
    """
    base = Image.open(rgb_image_path).convert("RGBA")
    bw, bh = base.size

    dst_mask = resample_bool_mask(
        mask_bool, mask_bbox_4326, rgb_bbox_4326, (bw, bh),
    )

    if outline_only:
        m = dst_mask.astype(np.uint8)
        edge_r = np.roll(m, 1, axis=0) != m
        edge_l = np.roll(m, -1, axis=0) != m
        edge_u = np.roll(m, 1, axis=1) != m
        edge_d = np.roll(m, -1, axis=1) != m
        paint = edge_r | edge_l | edge_u | edge_d
    else:
        paint = dst_mask

    overlay_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    overlay_arr[paint] = rgba
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)


def resample_bool_mask(
    mask_bool: np.ndarray,
    mask_bbox: tuple[float, float, float, float],
    base_bbox: tuple[float, float, float, float],
    base_size: tuple[int, int],
) -> np.ndarray:
    """Resample `mask_bool` (with `mask_bbox`) onto the base's pixel grid.

    If the bboxes match exactly, this just resizes via Pillow NEAREST. If
    they differ, the mask is sampled at each base pixel's lat/lon and
    out-of-bbox base pixels are set False.
    """
    bw, bh = base_size
    mh, mw = mask_bool.shape

    if mask_bbox == base_bbox:
        if (mw, mh) == (bw, bh):
            return mask_bool.astype(bool)
        pil = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
        pil = pil.resize((bw, bh), Image.NEAREST)
        return np.array(pil) > 0

    xmin_b, ymin_b, xmax_b, ymax_b = base_bbox
    xmin_m, ymin_m, xmax_m, ymax_m = mask_bbox

    xs = np.linspace(xmin_b, xmax_b, bw, endpoint=False) + (xmax_b - xmin_b) / (2 * bw)
    ys = np.linspace(ymax_b, ymin_b, bh, endpoint=False) - (ymax_b - ymin_b) / (2 * bh)

    mask_x_idx = ((xs - xmin_m) / (xmax_m - xmin_m) * mw).astype(np.int64)
    mask_y_idx = ((ymax_m - ys) / (ymax_m - ymin_m) * mh).astype(np.int64)

    mask_x_idx = np.clip(mask_x_idx, 0, mw - 1)
    mask_y_idx = np.clip(mask_y_idx, 0, mh - 1)

    ix, iy = np.meshgrid(mask_x_idx, mask_y_idx)
    resampled = mask_bool[iy, ix]

    out_of_bbox_x = (xs < xmin_m) | (xs > xmax_m)
    out_of_bbox_y = (ys < ymin_m) | (ys > ymax_m)
    resampled[:, out_of_bbox_x] = False
    resampled[out_of_bbox_y, :] = False
    return resampled
