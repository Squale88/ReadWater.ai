"""NDWI primitives + 4-band NAIP I/O — shared math library.

NOT DEPRECATED, despite naming overlap with the sunset NDWI-only water
mask approach. The NDWI primitives here are still the canonical math for
the new CV water-mask pipeline (see ``readwater.pipeline.cv.water_mask``,
which imports ``compute_ndwi`` and ``load_4band_tif`` for the NAIP carve
pass).

What IS deprecated: the workflow of using NDWI alone to produce a final
water mask. That approach over-claimed bright urban surfaces (roofs,
asphalt) as water and was replaced by the layered Google-styled-tile +
NAIP-carve + connectivity-filter pipeline. None of that workflow lives
in this file — this is just primitives.

If we ever rename this module to better reflect its role (e.g. to
``ndwi.py``), do so as a refactor with full call-site updates.

NDWI = (Green - NIR) / (Green + NIR)

Water absorbs NIR strongly and reflects green. Vegetation reflects NIR
strongly. So water pixels yield NDWI > ~0.1 and vegetation pixels yield
NDWI < 0. Sand and bare soil fall in between, depending on wetness.

Functions here are pure numpy — the rasterio dependency is isolated to the
load/save helpers at the bottom so the core NDWI math is testable without
geospatial setup.

Typical use (as building blocks for a larger water-mask pipeline):

    from readwater.api.data_sources.naip_4band import fetch_naip_4band, bbox_from_center
    from readwater.pipeline.water_mask import (
        load_4band_tif, compute_ndwi, threshold_water, save_mask_png,
    )

    result = fetch_naip_4band(bbox_from_center((26.011, -81.754), 16), "out/tile.tif")
    bands = load_4band_tif(result.path)
    ndwi = compute_ndwi(bands.green, bands.nir)
    mask = threshold_water(ndwi, threshold=0.1)
    # Don't ship `mask` as your final water mask; combine it with Google
    # styled tiles and a connectivity filter (see cv.water_mask) for
    # production use.

Dependencies: numpy always; rasterio only for the load/save helpers that
touch GeoTIFFs. Pure-numpy consumers can pass arrays directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class FourBandArrays:
    """Individual band arrays as float32 normalized [0, 1]."""

    red: np.ndarray      # (H, W) float32
    green: np.ndarray
    blue: np.ndarray
    nir: np.ndarray
    profile: dict | None = None  # rasterio profile for georeferenced save


# ---- NDWI and masking (pure numpy, no geo deps) ----


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index.

    Both inputs are float arrays scaled to the same range (typically [0, 1]
    or [0, 255]). Output is in [-1, 1]; water typically exceeds ~0.1.
    """
    g = green.astype(np.float32)
    n = nir.astype(np.float32)
    denom = g + n
    # Avoid divide-by-zero: set output to 0 where g + n == 0.
    out = np.zeros_like(g, dtype=np.float32)
    mask = denom > 0
    out[mask] = (g[mask] - n[mask]) / denom[mask]
    return out


def threshold_water(
    ndwi: np.ndarray,
    threshold: float = 0.0,
    min_run_pixels: int = 0,
) -> np.ndarray:
    """Binary water mask from an NDWI array.

    Args:
        ndwi: float array in ~[-1, 1].
        threshold: NDWI value above which a pixel is considered water.
            0.0 is a common strict threshold; 0.1 is more conservative.
        min_run_pixels: optional morphological cleanup — drop connected
            runs shorter than this. 0 disables.
    """
    mask = ndwi > threshold
    if min_run_pixels > 0:
        mask = _morph_open(mask, iters=min_run_pixels)
    return mask.astype(np.bool_)


def _shifted(m: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Shift mask by one pixel along axis, filling the exposed edge with False."""
    shifted = np.roll(m, direction, axis=axis)
    if axis == 0:
        edge = 0 if direction > 0 else -1
        shifted[edge, :] = False
    else:
        edge = 0 if direction > 0 else -1
        shifted[:, edge] = False
    return shifted


def _morph_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """4-connectivity erode-then-dilate. Pure numpy; fast at 1k resolution."""
    m = mask
    for _ in range(iters):
        up = _shifted(m, 0, -1)
        dn = _shifted(m, 0, 1)
        lt = _shifted(m, 1, -1)
        rt = _shifted(m, 1, 1)
        m = m & up & dn & lt & rt
    for _ in range(iters):
        up = _shifted(m, 0, -1)
        dn = _shifted(m, 0, 1)
        lt = _shifted(m, 1, -1)
        rt = _shifted(m, 1, 1)
        m = m | up | dn | lt | rt
    return m


# ---- PNG helpers (Pillow only) ----


def save_mask_png(mask: np.ndarray, out_path: str | Path) -> str:
    """Save a boolean mask as an 8-bit PNG (white = water, black = not)."""
    arr = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(arr, mode="L")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def save_mask_overlay_png(
    rgb_image_path: str | Path,
    mask: np.ndarray,
    out_path: str | Path,
    water_rgba: tuple[int, int, int, int] = (0, 150, 255, 110),
) -> str:
    """Pixel-aligned overlay. Assumes mask and base cover the SAME geography.

    Use ONLY when you know the mask's pixel grid matches the base image's
    geographic extent cell-for-cell. For cases where the mask covers a
    different or partial geographic area than the base, use
    `save_mask_overlay_png_georeferenced` instead.
    """
    base = Image.open(rgb_image_path).convert("RGBA")
    h, w = mask.shape
    if (w, h) != base.size:
        pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        pil_mask = pil_mask.resize(base.size, Image.NEAREST)
        mask = np.array(pil_mask) > 0

    overlay_arr = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    overlay_arr[mask] = water_rgba
    overlay = Image.fromarray(overlay_arr, mode="RGBA")

    out = Image.alpha_composite(base, overlay).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)


def save_mask_overlay_png_georeferenced(
    rgb_image_path: str | Path,
    mask_tif_path: str | Path,
    rgb_bbox_4326: tuple[float, float, float, float],
    out_path: str | Path,
    water_rgba: tuple[int, int, int, int] = (0, 150, 255, 110),
    no_data_outline: bool = True,
) -> str:
    """Overlay a georeferenced mask GeoTIFF onto a non-georeferenced RGB image,
    aligned via the RGB's known geographic bounding box.

    This is the correct way to combine a NAIP-derived mask (in some UTM zone,
    covering some partial area) with a separately-fetched RGB PNG (Google
    Static or USGS NAIP ImageServer, covering a specific lat/lon extent).

    Pixels with no source coverage are left untinted. When `no_data_outline`
    is True, the boundary between covered/uncovered regions is drawn so a
    human reviewer can tell at a glance which parts of the base had a mask
    to check against.

    Args:
        rgb_image_path: base image (PNG/JPEG, assumed covering `rgb_bbox_4326`).
        mask_tif_path: georeferenced 1-band mask GeoTIFF (nonzero = water).
        rgb_bbox_4326: (xmin, ymin, xmax, ymax) in WGS84 that rgb_image_path covers.
        out_path: output PNG path.
        water_rgba: tint color for water pixels.
        no_data_outline: draw boundary of coverage area.
    """
    _require_rasterio()
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    base = Image.open(rgb_image_path).convert("RGBA")
    bw, bh = base.size

    # Target grid: one pixel per base image pixel, CRS 4326.
    dst_transform = from_bounds(*rgb_bbox_4326, bw, bh)

    dst_mask = np.zeros((bh, bw), dtype=np.uint8)
    dst_covered = np.zeros((bh, bw), dtype=np.uint8)

    with rasterio.open(mask_tif_path) as src:
        # Reproject mask to target grid.
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        # Reproject a "fully 1" source to detect where we have coverage.
        src_ones = np.ones((src.height, src.width), dtype=np.uint8) * 255
        reproject(
            source=src_ones,
            destination=dst_covered,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )

    mask_bool = dst_mask > 0
    covered_bool = dst_covered > 0

    overlay_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    overlay_arr[mask_bool & covered_bool] = water_rgba

    if no_data_outline:
        # Mark the boundary of the covered region so the uncovered part is
        # visually distinguished from "covered but not water."
        covered_byte = covered_bool.astype(np.uint8)
        edge_r = np.roll(covered_byte, 1, axis=0) != covered_byte
        edge_l = np.roll(covered_byte, -1, axis=0) != covered_byte
        edge_u = np.roll(covered_byte, 1, axis=1) != covered_byte
        edge_d = np.roll(covered_byte, -1, axis=1) != covered_byte
        edge = edge_r | edge_l | edge_u | edge_d
        overlay_arr[edge] = (255, 255, 0, 200)  # yellow coverage boundary

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)


# ---- Rasterio-backed load/save (optional dep) ----


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "GeoTIFF helpers require the 'cv' extras. "
            "Install with: pip install -e '.[cv]'"
        ) from e


def load_4band_tif(
    tif_path: str | Path,
    nir_band_index: int = 4,
    red_band_index: int = 1,
    green_band_index: int = 2,
    blue_band_index: int = 3,
    normalize: bool = True,
) -> FourBandArrays:
    """Read a 4-band NAIP GeoTIFF, returning per-band arrays and the profile.

    Normalizes uint8 [0-255] or uint16 inputs to float32 [0, 1] when
    `normalize=True`.
    """
    _require_rasterio()
    import rasterio
    with rasterio.open(tif_path) as src:
        red = src.read(red_band_index)
        green = src.read(green_band_index)
        blue = src.read(blue_band_index)
        nir = src.read(nir_band_index)
        profile = src.profile.copy()

    def _norm(a):
        if not normalize:
            return a.astype(np.float32)
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        if a.dtype == np.uint16:
            return a.astype(np.float32) / 65535.0
        return a.astype(np.float32)

    return FourBandArrays(
        red=_norm(red),
        green=_norm(green),
        blue=_norm(blue),
        nir=_norm(nir),
        profile=profile,
    )


def save_mask_geotiff(
    mask: np.ndarray,
    profile: dict,
    out_path: str | Path,
) -> str:
    """Save a boolean mask as a 1-band uint8 GeoTIFF with the given profile.

    The profile should come from a FourBandArrays.profile (so CRS/transform
    line up with the source imagery).
    """
    _require_rasterio()
    import rasterio
    prof = dict(profile)
    prof.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write((mask.astype(np.uint8) * 255), indexes=1)
    return str(out_path)
