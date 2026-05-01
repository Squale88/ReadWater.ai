"""Fetch NAIP imagery into the per-cell cache.

For each requested cell, produces:
  - <cell>_naip_rgb.png      NAIP natural-color (USGS ImageServer, 1280x1280)
  - <cell>_naip_4band.tif    NAIP 4-band GeoTIFF (Planetary Computer)

Both files are consumed by downstream pipelines:
  - the NAIP RGB is the base image for the channel and habitat overlays
    (scripts/noaa_channel_mask.py, scripts/fwc_habitat_mask.py)
  - the 4-band TIF is read by the Google + NAIP hybrid water-mask
    pipeline (scripts/google_water_mask.py), which uses the NIR
    band-derived NDWI to carve smoothed-over islands out of Google's
    curated water polygons.

This script does NOT compute a water mask itself. The earlier NAIP-only
NDWI mask approach was abandoned in favor of the Google-anchored hybrid
because pixel-level NDWI on built-up shorelines false-positives on dark
roofs and fresh asphalt. See scripts/google_water_mask.py for the
production water-mask pipeline.

Usage:
  pip install -e '.[cv]'
  python scripts/fetch_naip_tifs.py --cell root-10-8
  python scripts/fetch_naip_tifs.py --cell all

The 4-band fetch requires the cv extras (rasterio + pystac-client +
planetary-computer). The RGB-only step works without them.
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
sys.path.insert(0, str(WORKTREE / "scripts"))

from _cells import CELLS  # noqa: E402
from readwater.api.providers.naip import NAIPProvider  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_naip"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


async def fetch_rgb(cell_id: str, center, zoom) -> str:
    provider = NAIPProvider()
    out_path = OUT_ROOT / f"{cell_id}_naip_rgb.png"
    print(f"[{cell_id}] fetching NAIP RGB -> {out_path}")
    await provider.fetch(center, zoom, str(out_path))
    return str(out_path)


def fetch_4band(cell_id: str, center, zoom) -> dict:
    try:
        from readwater.api.data_sources.naip_4band import (
            bbox_from_center, fetch_naip_4band,
        )
    except RuntimeError as e:
        print(f"[{cell_id}] skipping 4-band fetch: {e}")
        return {}

    bbox = bbox_from_center(center, zoom, image_size=640)
    tif_out = OUT_ROOT / f"{cell_id}_naip_4band.tif"
    print(f"[{cell_id}] fetching 4-band NAIP COG (mosaic) for bbox {bbox}")
    print(f"[{cell_id}]   -> {tif_out}")
    result = fetch_naip_4band(bbox, tif_out)
    print(f"[{cell_id}]   NAIP items merged: {result.item_id}")
    print(f"[{cell_id}]   year {result.acquired_year}, {result.bands} bands, "
          f"{result.height}x{result.width} px")
    return {
        "tif": result.path,
        "acquired_year": result.acquired_year,
    }


async def run_one(cell_id: str) -> dict:
    spec = CELLS[cell_id]
    center = spec["center"]
    zoom = spec["zoom"]
    print(f"\n=== {cell_id}  center={center}  zoom={zoom} ===")
    rgb_path = await fetch_rgb(cell_id, center, zoom)
    tif_info = fetch_4band(cell_id, center, zoom)
    return {"cell_id": cell_id, "rgb": rgb_path, **tif_info}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=list(CELLS.keys()) + ["all", "both"],
        default="root-10-8",
        help="Single cell id, or 'all' for every cell in _cells.CELLS. "
             "'both' is kept as a legacy alias meaning the two original cells.",
    )
    args = parser.parse_args()
    if args.cell == "all":
        cells = list(CELLS.keys())
    elif args.cell == "both":
        cells = ["root-10-8", "root-11-5"]
    else:
        cells = [args.cell]
    for cid in cells:
        await run_one(cid)
    print(f"\nAll artifacts in: {OUT_ROOT}")


if __name__ == "__main__":
    asyncio.run(main())
