"""Fetch FWC oyster reef + seagrass polygons and generate per-tile masks.

For each requested cell, produces:
  - <cell>_oyster_mask.png        : binary raster (white = surveyed oyster)
  - <cell>_oyster_mask.tif        : georeferenced GeoTIFF
  - <cell>_oyster_overlay.png     : NAIP RGB with oyster polygons tinted purple
  - <cell>_seagrass_mask.png      : binary raster (white = surveyed seagrass)
  - <cell>_seagrass_mask.tif      : georeferenced GeoTIFF
  - <cell>_seagrass_overlay.png   : NAIP RGB with seagrass polygons tinted green

Plus cached area-level GeoJSONs:
  - oyster_beds.geojson, seagrass.geojson — one per area, reused across cells.

Usage:
  pip install -e '.[cv]'
  python scripts/fwc_habitat_mask.py --cell both
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
from readwater.api.data_sources.fwc_habitats import (  # noqa: E402
    fetch_oyster_beds,
    fetch_seagrass,
)
from readwater.api.data_sources.naip_4band import bbox_from_center  # noqa: E402
from readwater.pipeline.polygon_mask import (  # noqa: E402
    rasterize_polygons,
    save_polygon_overlay_png,
)

OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_habitats"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Marco / Naples / Rookery Bay area bbox. Used once to pull the full FWC
# polygon set; per-cell masks are clipped from that.
AREA_BBOX = (-81.90, 25.85, -81.60, 26.30)

# Tint colors for the overlays. Distinct from channel (red) and water (blue).
OYSTER_RGBA = (180, 50, 200, 120)    # purple — marine GIS convention
SEAGRASS_RGBA = (50, 200, 50, 120)   # green — matches "vegetation" intuition


def ensure_oyster_geojson() -> str:
    out = OUT_ROOT / "oyster_beds.geojson"
    if out.exists():
        count = _count_features(out)
        print(f"using cached oyster GeoJSON ({count} features): {out}")
        return str(out)
    print(f"fetching FWC oyster beds for bbox {AREA_BBOX} -> {out}")
    result = fetch_oyster_beds(AREA_BBOX, out)
    print(f"  {result.feature_count} oyster reef polygons")
    return result.path


def ensure_seagrass_geojson() -> str:
    out = OUT_ROOT / "seagrass.geojson"
    if out.exists():
        count = _count_features(out)
        print(f"using cached seagrass GeoJSON ({count} features): {out}")
        return str(out)
    print(f"fetching FWC seagrass for bbox {AREA_BBOX} -> {out}")
    result = fetch_seagrass(AREA_BBOX, out)
    print(f"  {result.feature_count} seagrass polygons")
    return result.path


def _count_features(geojson_path: Path) -> int:
    with open(geojson_path, "r", encoding="utf-8") as f:
        return len(json.load(f).get("features", []))


def process_cell(
    cell_id: str,
    oyster_geojson: str,
    seagrass_geojson: str,
) -> dict:
    spec = CELLS[cell_id]
    center = spec["center"]
    zoom = spec["zoom"]
    bbox = bbox_from_center(center, zoom, image_size=640)
    print(f"\n=== {cell_id}  center={center}  zoom={zoom}  bbox={bbox} ===")

    out = {"cell_id": cell_id, "bbox": list(bbox)}
    naip_rgb = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_naip" / f"{cell_id}_naip_rgb.png"

    for kind, source_geojson, rgba in (
        ("oyster", oyster_geojson, OYSTER_RGBA),
        ("seagrass", seagrass_geojson, SEAGRASS_RGBA),
    ):
        mask_png = OUT_ROOT / f"{cell_id}_{kind}_mask.png"
        mask_tif = OUT_ROOT / f"{cell_id}_{kind}_mask.tif"
        raster_result = rasterize_polygons(
            source_geojson,
            bbox_4326=bbox,
            out_size=(1280, 1280),
            out_mask_png=mask_png,
            out_mask_tif=mask_tif,
        )
        print(f"  {kind:9s} fraction of tile: {raster_result.covered_fraction:.1%}")
        out[f"{kind}_fraction"] = float(raster_result.covered_fraction)
        out[f"{kind}_mask_png"] = str(mask_png)
        out[f"{kind}_mask_tif"] = str(mask_tif)

        overlay_path = OUT_ROOT / f"{cell_id}_{kind}_overlay.png"
        if naip_rgb.exists():
            mask_bool = np.array(Image.open(mask_png).convert("L")) > 0
            save_polygon_overlay_png(
                rgb_image_path=naip_rgb,
                mask_bool=mask_bool,
                rgb_bbox_4326=bbox,
                mask_bbox_4326=bbox,
                out_path=overlay_path,
                rgba=rgba,
                outline_only=False,
            )
            print(f"  {kind:9s} overlay -> {overlay_path}")
            out[f"{kind}_overlay"] = str(overlay_path)
        else:
            print(f"  skipping {kind} overlay: {naip_rgb} not found "
                  f"(run scripts/naip_water_mask.py first)")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=list(CELLS.keys()) + ["all", "both"],
        default="all",
        help="Single cell id, or 'all' for every cell in _cells.CELLS. "
             "'both' is kept as a legacy alias meaning the two original cells.",
    )
    args = parser.parse_args()

    oyster_gj = ensure_oyster_geojson()
    seagrass_gj = ensure_seagrass_geojson()

    if args.cell == "all":
        cells = list(CELLS.keys())
    elif args.cell == "both":
        cells = ["root-10-8", "root-11-5"]
    else:
        cells = [args.cell]
    results = []
    for cid in cells:
        results.append(process_cell(cid, oyster_gj, seagrass_gj))

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll artifacts in: {OUT_ROOT}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
