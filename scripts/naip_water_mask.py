"""Fetch NAIP imagery + generate an NDWI water mask for a test cell.

For each requested cell, produces:
  - <cell>_naip_rgb.png        : NAIP natural-color (USGS ImageServer, 1280x1280)
  - <cell>_naip_4band.tif      : NAIP 4-band GeoTIFF (Planetary Computer)
  - <cell>_water_mask.png      : binary NDWI water mask (white = water)
  - <cell>_water_overlay.png   : NAIP RGB with water pixels tinted blue

Usage:
  pip install -e '.[cv]'
  python scripts/naip_water_mask.py --cell root-10-8
  python scripts/naip_water_mask.py --cell both

Requires the cv extras (rasterio + pystac-client + planetary-computer) for
the water-mask step. The RGB-only fetch works without them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from readwater.api.providers.naip import NAIPProvider  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_naip"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CELLS = {
    "root-10-8": {"center": (26.011172, -81.753546), "zoom": 16},
    "root-11-5": {"center": (26.011172, -81.739780), "zoom": 16},
}


async def fetch_rgb(cell_id: str, center, zoom) -> str:
    provider = NAIPProvider()
    out_path = OUT_ROOT / f"{cell_id}_naip_rgb.png"
    print(f"[{cell_id}] fetching NAIP RGB -> {out_path}")
    await provider.fetch(center, zoom, str(out_path))
    return str(out_path)


def fetch_4band_and_mask(cell_id: str, center, zoom) -> dict:
    try:
        from readwater.api.data_sources.naip_4band import (
            bbox_from_center, fetch_naip_4band,
        )
        from readwater.pipeline.water_mask import (
            compute_ndwi,
            load_4band_tif,
            save_mask_geotiff,
            save_mask_overlay_png_georeferenced,
            save_mask_png,
            threshold_water,
        )
    except RuntimeError as e:
        print(f"[{cell_id}] skipping water mask: {e}")
        return {}

    bbox = bbox_from_center(center, zoom, image_size=640)
    tif_out = OUT_ROOT / f"{cell_id}_naip_4band.tif"
    print(f"[{cell_id}] fetching 4-band NAIP COG (mosaic) for bbox {bbox}")
    print(f"[{cell_id}]   -> {tif_out}")
    result = fetch_naip_4band(bbox, tif_out)
    print(f"[{cell_id}]   NAIP items merged: {result.item_id}")
    print(f"[{cell_id}]   year {result.acquired_year}, {result.bands} bands, "
          f"{result.height}x{result.width} px")

    print(f"[{cell_id}] computing NDWI...")
    bands = load_4band_tif(result.path, nir_band_index=result.nir_band_index)
    ndwi = compute_ndwi(bands.green, bands.nir)
    mask = threshold_water(ndwi, threshold=0.0, min_run_pixels=2)
    print(f"[{cell_id}]   water fraction: {mask.mean():.1%}")

    # Save the mask both as a simple PNG and as a georeferenced GeoTIFF.
    mask_png = OUT_ROOT / f"{cell_id}_water_mask.png"
    save_mask_png(mask, mask_png)
    mask_tif = OUT_ROOT / f"{cell_id}_water_mask.tif"
    save_mask_geotiff(mask, bands.profile, mask_tif)
    print(f"[{cell_id}]   mask PNG -> {mask_png}")
    print(f"[{cell_id}]   mask GeoTIFF (CRS-aware) -> {mask_tif}")

    # Overlay: use the georeferenced alignment so the mask sits on the RGB
    # in the right place even when the 4-band coverage is a subset of the tile.
    overlay_path = OUT_ROOT / f"{cell_id}_water_overlay.png"
    rgb_path = OUT_ROOT / f"{cell_id}_naip_rgb.png"
    if rgb_path.exists():
        save_mask_overlay_png_georeferenced(
            rgb_image_path=rgb_path,
            mask_tif_path=mask_tif,
            rgb_bbox_4326=bbox,
            out_path=overlay_path,
        )
        print(f"[{cell_id}]   georeferenced overlay -> {overlay_path}")

    return {
        "rgb": str(rgb_path),
        "tif": result.path,
        "mask_png": str(mask_png),
        "mask_tif": str(mask_tif),
        "overlay": str(overlay_path),
        "water_fraction": float(mask.mean()),
        "acquired_year": result.acquired_year,
    }


async def run_one(cell_id: str) -> dict:
    spec = CELLS[cell_id]
    center = spec["center"]
    zoom = spec["zoom"]
    print(f"\n=== {cell_id}  center={center}  zoom={zoom} ===")
    rgb_path = await fetch_rgb(cell_id, center, zoom)
    mask_info = fetch_4band_and_mask(cell_id, center, zoom)
    return {"cell_id": cell_id, "rgb": rgb_path, **mask_info}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=list(CELLS.keys()) + ["both"], default="root-10-8")
    args = parser.parse_args()
    cells = list(CELLS.keys()) if args.cell == "both" else [args.cell]
    for cid in cells:
        await run_one(cid)
    print(f"\nAll artifacts in: {OUT_ROOT}")


if __name__ == "__main__":
    asyncio.run(main())
