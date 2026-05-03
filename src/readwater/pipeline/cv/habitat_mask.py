"""Fetch FWC oyster reef + seagrass polygons and generate per-tile masks.

For each requested cell, produces under ``data/areas/<area>/masks/<kind>/``
(resolved through ``readwater.storage``):

  - <cell>_oyster_mask.png        : binary raster (white = surveyed oyster)
  - <cell>_oyster_mask.tif        : georeferenced GeoTIFF
  - <cell>_oyster_overlay.png     : NAIP RGB with oyster polygons tinted purple
  - <cell>_seagrass_mask.png      : binary raster (white = surveyed seagrass)
  - <cell>_seagrass_mask.tif      : georeferenced GeoTIFF
  - <cell>_seagrass_overlay.png   : NAIP RGB with seagrass polygons tinted green

Plus area-level cached GeoJSONs at ``data/areas/<area>/masks/``:

  - oyster_beds.geojson, seagrass.geojson — one per area, reused across cells.

Cell registry: cell ids and centers come from the area's manifest via
``readwater.areas.Area``. The legacy ``scripts/_cells.py`` dependency and
the ``--cell both`` shorthand alias are gone; use ``--cell all`` or repeat
``--cell`` explicitly.

Usage (via shim):
  pip install -e '.[cv]'
  python scripts/fwc_habitat_mask.py --cell root-10-8
  python scripts/fwc_habitat_mask.py --cell all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from readwater import storage
from readwater.api.data_sources.fwc_habitats import (
    fetch_oyster_beds,
    fetch_seagrass,
)
from readwater.api.data_sources.naip_4band import bbox_from_center
from readwater.areas import Area
from readwater.pipeline.polygon_mask import (
    rasterize_polygons,
    save_polygon_overlay_png,
)

# Marco / Naples / Rookery Bay area bbox. Used once to pull the full FWC
# polygon set; per-cell masks are clipped from that.
AREA_BBOX = (-81.90, 25.85, -81.60, 26.30)

# Tint colors for the overlays. Distinct from channel (red) and water (blue).
OYSTER_RGBA = (180, 50, 200, 120)    # purple — marine GIS convention
SEAGRASS_RGBA = (50, 200, 50, 120)   # green — matches "vegetation" intuition

# Per-cell mask raster size (matches the legacy 1280x1280 grid the rest of
# the pipeline uses).
MASK_SIZE = (1280, 1280)


def ensure_oyster_geojson(area_id: str) -> str:
    out = storage.oyster_beds_geojson_path(area_id)
    if out.exists():
        count = _count_features(out)
        print(f"using cached oyster GeoJSON ({count} features): {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching FWC oyster beds for bbox {AREA_BBOX} -> {out}")
    result = fetch_oyster_beds(AREA_BBOX, out)
    print(f"  {result.feature_count} oyster reef polygons")
    return result.path


def ensure_seagrass_geojson(area_id: str) -> str:
    out = storage.seagrass_geojson_path(area_id)
    if out.exists():
        count = _count_features(out)
        print(f"using cached seagrass GeoJSON ({count} features): {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching FWC seagrass for bbox {AREA_BBOX} -> {out}")
    result = fetch_seagrass(AREA_BBOX, out)
    print(f"  {result.feature_count} seagrass polygons")
    return result.path


def _count_features(geojson_path: Path) -> int:
    with open(geojson_path, "r", encoding="utf-8") as f:
        return len(json.load(f).get("features", []))


def process_cell(
    area_id: str,
    cell_id: str,
    oyster_geojson: str,
    seagrass_geojson: str,
) -> dict:
    area = Area(area_id)
    if not area.has_cell(cell_id):
        raise SystemExit(
            f"unknown cell {cell_id!r} for area {area_id!r}; "
            f"see Area({area_id!r}).cell_ids()"
        )
    cell = area.cell(cell_id)
    center = cell.center
    if center is None:
        raise SystemExit(
            f"cell {cell_id!r} has no center in the manifest; "
            f"rebuild manifest with scripts/build_manifest.py"
        )
    # Every v1 cell is a z16 tile; that's invariant for the project.
    zoom = 16
    bbox = bbox_from_center(center, zoom, image_size=640)
    print(f"\n=== {cell_id}  center={center}  zoom={zoom}  bbox={bbox} ===")

    out: dict = {"cell_id": cell_id, "bbox": list(bbox)}
    # TODO: NAIP RGB tiles still live in the legacy sibling directory
    # ``data/areas/<area>_naip``. Consolidating them under
    # ``area_root() / "naip" / ...`` is deferred to a future PR — they're
    # fetched by scripts/fetch_naip_tifs.py which hasn't been migrated yet.
    naip_rgb = (
        storage.data_root() / "areas" / f"{area_id}_naip"
        / f"{cell_id}_naip_rgb.png"
    )

    for kind, source_geojson, rgba in (
        ("oyster", oyster_geojson, OYSTER_RGBA),
        ("seagrass", seagrass_geojson, SEAGRASS_RGBA),
    ):
        if kind == "oyster":
            mask_png = storage.oyster_mask_path(area_id, cell_id)
        else:
            mask_png = storage.seagrass_mask_path(area_id, cell_id)
        mask_tif = mask_png.with_suffix(".tif")

        # rasterize_polygons writes its own PNG/TIF directly. The PNG path
        # is non-atomic by design here — the rasterio TIF write next to it
        # cannot be made atomic via tempfile-rename (rasterio opens the
        # destination path), so for symmetry we leave the PNG as a direct
        # write too. Acceptable for derived data: a partial file gets
        # rewritten on the next run.
        # rasterio writes in-place; not atomic. Acceptable for derived data.
        mask_png.parent.mkdir(parents=True, exist_ok=True)
        raster_result = rasterize_polygons(
            source_geojson,
            bbox_4326=bbox,
            out_size=MASK_SIZE,
            out_mask_png=mask_png,
            out_mask_tif=mask_tif,
        )
        print(f"  {kind:9s} fraction of tile: {raster_result.covered_fraction:.1%}")
        out[f"{kind}_fraction"] = float(raster_result.covered_fraction)
        out[f"{kind}_mask_png"] = str(mask_png)
        out[f"{kind}_mask_tif"] = str(mask_tif)

        overlay_path = mask_png.with_name(f"{cell_id}_{kind}_overlay.png")
        if naip_rgb.exists():
            mask_bool = np.array(Image.open(mask_png).convert("L")) > 0
            # save_polygon_overlay_png writes the composed PNG directly;
            # not atomic, same justification as above.
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
                  f"(run scripts/fetch_naip_tifs.py first)")

    return out


def _resolve_cells(area: Area, requested: list[str]) -> list[str]:
    """Expand 'all' and validate every requested cell against the area."""
    if not requested:
        return area.cell_ids()
    if "all" in requested:
        if len(requested) != 1:
            raise SystemExit("--cell all cannot be combined with other --cell values")
        return area.cell_ids()
    bad = [c for c in requested if not area.has_cell(c)]
    if bad:
        raise SystemExit(
            f"unknown cell(s) for area {area.area_id!r}: {bad!r}. "
            f"See Area({area.area_id!r}).cell_ids() for the full list."
        )
    return requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area", default="rookery_bay_v2",
        help="Area id (default: rookery_bay_v2).",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Cell id like root-10-8. Pass 'all' for every cell in the area, "
             "or repeat the flag for multiple cells. Defaults to 'all'.",
    )
    args = parser.parse_args()

    area = Area(args.area)
    cells = _resolve_cells(area, args.cell)

    oyster_gj = ensure_oyster_geojson(args.area)
    seagrass_gj = ensure_seagrass_geojson(args.area)

    results = []
    for cid in cells:
        results.append(process_cell(args.area, cid, oyster_gj, seagrass_gj))

    summary_path = storage.masks_root(args.area) / "habitat_summary.json"
    storage.atomic_write_json(summary_path, results)
    print(f"\nAll artifacts in: {storage.masks_root(args.area)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
