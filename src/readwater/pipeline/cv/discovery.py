"""Deterministic CV-based recursive discovery for an area.

Replaces the LLM-driven dual_pass_grid_scoring in cell_analyzer with a
geometric water-mask rubric. At each level of the recursion (z12 -> z14
-> z16), we evaluate 16 sub-cells of the current parent and keep only
those that pass:

  Rubric per sub-cell:
    1. connected_water_pct >= MIN_CONNECTED_WATER_PCT (~no meaningful water)
    2. widest_water_m       >= MIN_WIDEST_WATER_M     (~drainage-canal sized only)

The connectivity test mirrors the per-cell water-mask pipeline at the
appropriate scale: for evaluating sub-cells at zoom Z (i.e. parent at
Z-2), we fetch a detail tile at (Z-2) and an isolation tile at (Z-3),
then `final_ocean = perim_connected(detail) AND NOT isolated(isolation)`.

Evaluation specifics per zoom step:
  - Parent z12 (root area) -> evaluate 16 z14 sub-cells with z12+z11 tiles.
  - Parent z14            -> evaluate 16 z16 sub-cells with z14+z13 tiles.
  - z16 is terminal — we don't descend further (matches TERMINAL_ZOOM in
    cell_analyzer).

Validated by the same-area manual review on rookery_bay_v2 root-7 + root-2
(zero false positives, zero false negatives against user truth).

This module is the standalone C-step. The B-step (replacing the LLM
scoring inside cell_analyzer) wraps these functions into the existing
discovery-recursion path.

Usage:
  python scripts/cv_discover.py --area rookery_bay_v2

Output:
  data/areas/<area>/cv_discovery_<ts>.json   (the kept cells + per-step trace)
  data/areas/<area>/_discovery_cache/        (fetched styled tiles, reusable)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from readwater import storage
from readwater.api.data_sources.naip_4band import bbox_from_center
from readwater.pipeline.cv.helpers import connected_components, erode_4conn
from readwater.pipeline.cv.water_mask import (
    crop_wide_mask_to_cell,
    fetch_styled_water,
    perimeter_connected_mask,
    water_mask_from_styled,
)

# ---- Rubric (two-tier: looser at coarse levels, stricter at fine levels) ----

# Always required: cell must have meaningful land mass (rejects open ocean)
# and meaningful ocean-connected water (rejects pure inland).
MIN_LAND_PCT            = 3.0   # %  cell must have at least this much LAND
                                # (else it's open ocean, no fishable structure)

# Per-level water-percent threshold. Level-1 evaluation runs at coarse
# resolution (z12 detail = ~17 m/px) so we set a lower floor; level-2
# runs at z14 detail (~9.5 m/px) and can afford to be stricter.
MIN_CONNECTED_WATER_PCT_L1 = 1.0   # %  level-1 (evaluating z14 sub-cells)
MIN_CONNECTED_WATER_PCT_L2 = 2.0   # %  level-2 (evaluating z16 sub-cells)

# Widest-channel filter only applied at level-2. At level-1 the source
# resolution (z12 = 17 m/px) is too coarse to reliably measure 50 m
# channels — root-12 (real Marco Island fishery) gets dropped if we
# require widest>=50m at level-1.
MIN_WIDEST_WATER_M_L2   = 50.0  # m  rejects drainage-canal-only cells

SECTIONS                = 4     # 4x4 sub-grid per parent (matches discovery)
TERMINAL_ZOOM           = 16    # match cell_analyzer.TERMINAL_ZOOM
EVAL_CROP_RES_PX        = 256   # all sub-cell crops resampled to 256x256

# Approx miles-per-degree-latitude for converting bbox extent to meters.
_M_PER_DEG_LAT = 111_000


# ---- Result shape ----


@dataclass
class SubcellEval:
    cell_num: int                # 1..16 within parent
    kept: bool
    water_pct: float             # ocean-connected water as % of cell
    land_pct: float              # land as % of cell (raw water mask)
    widest_m: float              # widest CC width in meters
    n_ccs: int                   # number of ocean-connected CCs >=10 px
    drop_reason: str = ""        # explanation when kept=False


@dataclass
class DiscoveryResult:
    area_id: str
    root_bbox: dict
    rubric: dict = field(default_factory=lambda: {
        "min_land_pct": MIN_LAND_PCT,
        "min_connected_water_pct_l1": MIN_CONNECTED_WATER_PCT_L1,
        "min_connected_water_pct_l2": MIN_CONNECTED_WATER_PCT_L2,
        "min_widest_water_m_l2": MIN_WIDEST_WATER_M_L2,
    })
    # Per-parent evaluation trace, keyed by parent label ("root", "root-2", ...).
    # Each value is a list of 16 SubcellEval dicts.
    evaluations: dict[str, list[dict]] = field(default_factory=dict)
    kept_z16_cells: list[str] = field(default_factory=list)
    timestamp: str = ""


# ---- Geometry helpers ----


def _bbox_center(bb: dict) -> tuple[float, float]:
    return ((bb["north"] + bb["south"]) / 2,
            (bb["east"] + bb["west"]) / 2)


def _subcell_bbox(parent_bb: dict, cell_num: int) -> dict:
    row = (cell_num - 1) // SECTIONS
    col = (cell_num - 1) % SECTIONS
    h = (parent_bb["north"] - parent_bb["south"]) / SECTIONS
    w = (parent_bb["east"]  - parent_bb["west"])  / SECTIONS
    return {
        "north": parent_bb["north"] - row * h,
        "south": parent_bb["north"] - (row + 1) * h,
        "west":  parent_bb["west"]  + col * w,
        "east":  parent_bb["west"]  + (col + 1) * w,
    }


def _bbox_to_lon_lat_tuple(bb: dict) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for crop_wide_mask_to_cell."""
    return (bb["west"], bb["south"], bb["east"], bb["north"])


def _cell_extent_m(bb: dict) -> float:
    """Approximate cell width in meters (lat extent × m/deg)."""
    return (bb["north"] - bb["south"]) * _M_PER_DEG_LAT


# ---- CC width measurement ----


def _widest_width_px(cc: dict, cap: int = 100) -> int:
    """Widest width (in px) of a connected component, via iterative erosion.

    Inscribed circle radius = max iterations of 4-connected erosion before
    the CC vanishes. Width = 2 * radius. Cap at 100 iterations to avoid
    runaway on any pathological large CC (200 px > any real channel at the
    scales we evaluate).
    """
    x0, y0, x1, y1 = cc["bbox"]
    bw, bh = x1 - x0, y1 - y0
    local = np.zeros((bh, bw), dtype=bool)
    for (y, x) in cc["pixels"]:
        local[y - y0, x - x0] = True
    cur = local.copy()
    for it in range(1, cap + 1):
        nxt = erode_4conn(cur, 1)
        if not nxt.any():
            return it * 2
        cur = nxt
    return cap * 2


# ---- Tile fetch + connectivity ----


def _ensure_tile(center: tuple[float, float], zoom: int,
                 cache_dir: Path, name: str) -> Path:
    """Fetch a styled water tile to cache_dir if not present, return path."""
    out_path = cache_dir / f"{name}_z{zoom}_styled.png"
    if not out_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        fetch_styled_water(center, zoom, out_path)
    return out_path


def _final_ocean_mask(parent_bb: dict, parent_zoom: int,
                       cache_dir: Path, label: str
                       ) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Build the final ocean-connected mask covering this parent's area.

    Returns (final_ocean_mask, raw_detail_water_mask, detail_tile_bbox_lonlat).

    Approach mirrors gulf_connected_mask_for_cell at the discovery scale:
      detail_zoom    = parent_zoom        (4x parent extent)
      isolation_zoom = parent_zoom - 1    (8x parent extent; perimeter sits
                                           well past any inland water bodies)
    """
    center = _bbox_center(parent_bb)
    detail_zoom = parent_zoom
    isolation_zoom = parent_zoom - 1

    detail_tile = _ensure_tile(center, detail_zoom, cache_dir, f"{label}_detail")
    isolation_tile = _ensure_tile(center, isolation_zoom, cache_dir, f"{label}_isolation")

    detail_water = water_mask_from_styled(detail_tile)
    detail_perim = perimeter_connected_mask(detail_water, bridge_dilate_iters=1)
    isolation_water = water_mask_from_styled(isolation_tile)
    isolation_perim = perimeter_connected_mask(isolation_water, bridge_dilate_iters=1)
    isolation_isolated = isolation_water & ~isolation_perim

    detail_bb = bbox_from_center(center, detail_zoom, image_size=640)
    isolation_bb = bbox_from_center(center, isolation_zoom, image_size=640)
    iso_in_detail = crop_wide_mask_to_cell(
        isolation_isolated, isolation_bb, detail_bb, detail_water.shape,
    )
    final_ocean = detail_perim & ~iso_in_detail
    return final_ocean, detail_water, detail_bb


# ---- Sub-cell evaluation ----


def evaluate_subcells(parent_bb: dict, parent_zoom: int,
                      cache_dir: Path, label: str) -> list[SubcellEval]:
    """Apply the level-appropriate rubric to each of the 16 sub-cells.

    Rubric (per sub-cell):
      DROP if land_pct < MIN_LAND_PCT                 -> open ocean
      DROP if water_pct < MIN_CONNECTED_WATER_PCT_L*  -> no inshore water
      LEVEL 2 ONLY: DROP if widest_m < MIN_WIDEST_WATER_M_L2  -> drainage-canal sized

    The widest-channel filter is only applied at level-2 (parent_zoom=14,
    evaluating z16 sub-cells). At level-1 (parent_zoom=12, evaluating z14
    sub-cells) the source resolution is too coarse to reliably measure
    50 m channels — we'd false-drop cells like root-12 (Marco Island
    fishery) whose channels are real but sub-resolved at z12.
    """
    final_ocean, raw_water, detail_bb = _final_ocean_mask(
        parent_bb, parent_zoom, cache_dir, label,
    )

    # Decide which level we're at and pick thresholds.
    is_level2 = (parent_zoom == 14)   # evaluating z16 sub-cells
    min_water_pct = (MIN_CONNECTED_WATER_PCT_L2 if is_level2
                     else MIN_CONNECTED_WATER_PCT_L1)

    # Per-cell extent at the SUB-cell scale (parent / 4 on each side)
    subcell_extent_m = _cell_extent_m(parent_bb) / SECTIONS
    px_to_m = subcell_extent_m / EVAL_CROP_RES_PX

    out: list[SubcellEval] = []
    for sub_n in range(1, SECTIONS * SECTIONS + 1):
        sb = _subcell_bbox(parent_bb, sub_n)
        sb_tuple = _bbox_to_lon_lat_tuple(sb)
        cell_ocean = crop_wide_mask_to_cell(
            final_ocean, detail_bb, sb_tuple,
            (EVAL_CROP_RES_PX, EVAL_CROP_RES_PX),
        )
        cell_raw_water = crop_wide_mask_to_cell(
            raw_water, detail_bb, sb_tuple,
            (EVAL_CROP_RES_PX, EVAL_CROP_RES_PX),
        )
        water_pct = 100 * cell_ocean.mean()
        land_pct  = 100 * (~cell_raw_water).mean()
        ccs = connected_components(cell_ocean, min_pixels=10)
        widest_m = (max(_widest_width_px(c) for c in ccs) * px_to_m
                    if ccs else 0.0)

        # Apply rubric in priority order so drop_reason is informative
        if land_pct < MIN_LAND_PCT:
            kept, reason = False, f"open ocean (land<{MIN_LAND_PCT}%)"
        elif water_pct < min_water_pct:
            kept, reason = False, f"no inshore water (<{min_water_pct}%)"
        elif is_level2 and widest_m < MIN_WIDEST_WATER_M_L2:
            kept, reason = False, f"canal-only (widest<{MIN_WIDEST_WATER_M_L2:.0f}m)"
        else:
            kept, reason = True, ""

        out.append(SubcellEval(
            cell_num=sub_n,
            kept=kept,
            water_pct=round(water_pct, 2),
            land_pct=round(land_pct, 2),
            widest_m=round(widest_m, 1),
            n_ccs=len(ccs),
            drop_reason=reason,
        ))
    return out


# ---- Top-level discovery ----


def _load_root_bbox(area_id: str) -> dict:
    """Read the area's root z12 bbox from its discovery metadata.json."""
    metadata_path = storage.area_root(area_id) / "images" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata.json at {metadata_path}; can't bootstrap discovery."
        )
    md = json.loads(metadata_path.read_text(encoding="utf-8"))
    for entry in md:
        if entry.get("cell_id") == "root" and entry.get("depth") == 0:
            return entry["bbox"]
    raise ValueError(f"No depth-0 root entry in {metadata_path}")


def discover_area(area_id: str, cache_dir: Path | None = None,
                   logger=None) -> DiscoveryResult:
    """Recursively discover all z16 cells in an area using the deterministic
    rubric. Returns a DiscoveryResult with the kept cells and per-step
    evaluation trace.
    """
    log = logger or print
    if cache_dir is None:
        cache_dir = storage.area_root(area_id) / "_discovery_cache"

    root_bbox = _load_root_bbox(area_id)
    result = DiscoveryResult(area_id=area_id, root_bbox=root_bbox)

    # Level 1: 16 z14 sub-cells of the root z12 area
    log(f"[Level 1] Evaluating root's 16 z14 sub-cells "
        f"(detail z12, isolation z11)...")
    level1 = evaluate_subcells(root_bbox, 12, cache_dir, "root")
    result.evaluations["root"] = [vars(r) for r in level1]
    kept_z14 = [r.cell_num for r in level1 if r.kept]
    log(f"  z14 cells kept: {kept_z14}")

    # Level 2: for each kept z14, evaluate its 16 z16 sub-cells
    log(f"[Level 2] Evaluating z16 sub-cells of {len(kept_z14)} kept z14 parents...")
    for z14_n in kept_z14:
        z14_bb = _subcell_bbox(root_bbox, z14_n)
        label = f"root-{z14_n}"
        log(f"  {label}: ", end="")
        level2 = evaluate_subcells(z14_bb, 14, cache_dir, label)
        result.evaluations[label] = [vars(r) for r in level2]
        kept_z16 = [r.cell_num for r in level2 if r.kept]
        log(f"kept {kept_z16}")
        for z16_n in kept_z16:
            result.kept_z16_cells.append(f"root-{z14_n}-{z16_n}")

    result.kept_z16_cells.sort()
    result.timestamp = datetime.now(timezone.utc).isoformat()
    return result


def write_result(result: DiscoveryResult, output_path: Path) -> Path:
    """Atomic-write the discovery result as JSON."""
    payload = {
        "area_id": result.area_id,
        "timestamp": result.timestamp,
        "root_bbox": result.root_bbox,
        "rubric": result.rubric,
        "kept_z16_count": len(result.kept_z16_cells),
        "kept_z16_cells": result.kept_z16_cells,
        "evaluations": result.evaluations,
    }
    storage.atomic_write_json(output_path, payload)
    return output_path
