"""Fetch NOAA ENC chart + generate charted-channel mask for Marco/Naples tiles.

For each requested cell, produces:
  - ENC download (cached across runs) covering the area
  - <cell>_channels.geojson     : FAIRWY + DRGARE polygons plus a thin
                                  marker-derived channel-lane indicator
                                  built by joining midpoints of the
                                  numbered BCNLAT/BOYLAT lateral markers
  - <cell>_channel_mask.png     : binary channel raster (white = charted channel)
  - <cell>_channel_mask.tif     : georeferenced GeoTIFF version
  - <cell>_channel_overlay.png  : NAIP RGB base with channel polygons tinted red

Chart selection: US4FL1JT (Gordon Pass to Gullivan Bay, 1:45,000) covers
Rookery Bay, Marco Island, and Naples. For other areas, pass --chart-id.

Usage:
  pip install -e '.[cv]'
  python scripts/noaa_channel_mask.py --cell root-10-8
  python scripts/noaa_channel_mask.py --cell all --chart-id US4FL1JT
"""

from __future__ import annotations

import argparse
import json
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

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from _cells import CELLS  # noqa: E402
from readwater.api.data_sources.naip_4band import bbox_from_center  # noqa: E402
from readwater.api.data_sources.noaa_enc import (  # noqa: E402
    download_enc,
    extract_channels,
)
from readwater.pipeline.polygon_mask import (  # noqa: E402
    rasterize_polygons,
    save_polygon_overlay_png,
)

# Channel overlay tint (red, for "warning — this is a boating channel, do
# not label it a drain").
CHANNEL_RGBA = (255, 0, 0, 120)

OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_channels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
ENC_CACHE = REPO_ROOT / "data" / "noaa_enc_cache"
ENC_CACHE.mkdir(parents=True, exist_ok=True)

# US4FL1JT: "Gordon Pass to Gullivan Bay" (1:45,000) — covers Rookery Bay,
# Marco Island, Naples. Identified by querying the NOAA ENC product catalog
# for charts whose polygon contains the test-cell center (26.011, -81.754).
DEFAULT_CHART_ID = "US4FL1JT"

# Rough bbox (4326) covering Marco/Naples/Rookery Bay — used to clip
# features during extraction so we don't hold the whole chart's vector
# data in memory.
AREA_BBOX = (-81.90, 25.85, -81.60, 26.30)


def ensure_chart_extracted(chart_id: str) -> str:
    """Download + extract the ENC chart if not already cached. Returns .000 path."""
    result = download_enc(chart_id, ENC_CACHE)
    return result.enc_file


def ensure_channels_geojson(enc_file: str, chart_id: str) -> str:
    """Extract channels once per chart and cache as GeoJSON."""
    out_path = OUT_ROOT / f"{chart_id}_channels.geojson"
    if out_path.exists():
        print(f"using cached channels GeoJSON: {out_path}")
        return str(out_path)

    print(f"extracting channels from {enc_file} -> {out_path}")
    # Default extract_channels behavior: FAIRWY + DRGARE polygons plus
    # marker-derived channel-lane indicators (BCNLAT/BOYLAT lateral
    # markers, paired by sequence number, midpoints joined as a thin
    # 10 m-wide centerline). SEAARE / DEPARE are off by default — see
    # noaa_enc.extract_channels for opt-in flags.
    result = extract_channels(
        enc_file,
        out_path,
        bbox_4326=AREA_BBOX,
    )
    print(f"  features: {result.feature_count}  "
          f"counts: {result.counts_by_class}  "
          f"clipped_out: {result.clipped_out}")
    return str(out_path)


def run_one(cell_id: str, chart_id: str, channels_geojson: str) -> dict:
    spec = CELLS[cell_id]
    center = spec["center"]
    zoom = spec["zoom"]
    bbox = bbox_from_center(center, zoom, image_size=640)
    print(f"\n=== {cell_id}  center={center}  zoom={zoom}  bbox={bbox} ===")

    mask_png = OUT_ROOT / f"{cell_id}_channel_mask.png"
    mask_tif = OUT_ROOT / f"{cell_id}_channel_mask.tif"
    raster_result = rasterize_polygons(
        channels_geojson,
        bbox_4326=bbox,
        out_size=(1280, 1280),
        out_mask_png=mask_png,
        out_mask_tif=mask_tif,
    )
    print(f"  channel fraction of tile: {raster_result.covered_fraction:.1%}")
    print(f"  mask PNG -> {mask_png}")
    print(f"  mask GeoTIFF -> {mask_tif}")

    # Overlay onto the NAIP RGB we produced in the previous step. Falls back
    # gracefully if that file doesn't exist yet.
    naip_rgb = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_naip" / f"{cell_id}_naip_rgb.png"
    overlay_path = OUT_ROOT / f"{cell_id}_channel_overlay.png"
    if naip_rgb.exists():
        mask_bool = np.array(Image.open(mask_png).convert("L")) > 0
        save_polygon_overlay_png(
            rgb_image_path=naip_rgb,
            mask_bool=mask_bool,
            rgb_bbox_4326=bbox,
            mask_bbox_4326=bbox,
            out_path=overlay_path,
            rgba=CHANNEL_RGBA,
            outline_only=False,
        )
        print(f"  overlay (NAIP + channel tint) -> {overlay_path}")
    else:
        print(f"  skipping overlay: {naip_rgb} not found "
              f"(run scripts/fetch_naip_tifs.py first)")

    return {
        "cell_id": cell_id,
        "bbox": list(bbox),
        "channel_fraction": float(raster_result.covered_fraction),
        "mask_png": str(mask_png),
        "mask_tif": str(mask_tif),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=list(CELLS.keys()) + ["all", "both"],
        default="all",
        help="Single cell id, or 'all' for every cell in _cells.CELLS. "
             "'both' is kept as a legacy alias meaning the two original cells.",
    )
    parser.add_argument("--chart-id", default=DEFAULT_CHART_ID)
    args = parser.parse_args()

    print(f"Chart: {args.chart_id}")
    enc_file = ensure_chart_extracted(args.chart_id)
    print(f"ENC .000 path: {enc_file}")

    channels_geojson = ensure_channels_geojson(enc_file, args.chart_id)

    if args.cell == "all":
        cells = list(CELLS.keys())
    elif args.cell == "both":
        cells = ["root-10-8", "root-11-5"]
    else:
        cells = [args.cell]
    results = []
    for cid in cells:
        results.append(run_one(cid, args.chart_id, channels_geojson))

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll artifacts in: {OUT_ROOT}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
