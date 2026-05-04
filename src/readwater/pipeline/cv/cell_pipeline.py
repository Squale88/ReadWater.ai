"""Per-cell CV pipeline runner — shared between run_area.py and cell_analyzer.py.

Single source of truth for "run the full CV pipeline on one cell": ensure
water mask, ensure habitat masks, run all 4 detectors, run the orchestrator.
Both the batch entry point (``run_area.py``) and the discovery integration
(``cell_analyzer.py``) call into ``run_cell`` here.

Two convenience flavors:

  - ``run_cell(area_id, cell, oyster_geojson, seagrass_geojson, ...)``
    Used by run_area, which already has an Area + Cell loaded and has
    already cached the area's habitat geojsons.

  - ``run_cell_full(area_id, cell_id, ...)``
    Used by cell_analyzer, which has only an area_id + cell_id string.
    Resolves the Area, ensures the habitat geojsons are cached, then
    delegates to ``run_cell``. Convenient for one-shot callers; slightly
    more setup overhead per call than the explicit form.

The CellResult dataclass is the typed return: cell_id, duration, status,
captured log, failure metadata.
"""

from __future__ import annotations

import io
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass

from readwater.areas import Area, Cell
from readwater.pipeline.cv import (
    detect_drains,
    detect_islands,
    detect_pockets,
    detect_points,
    habitat_mask,
    orchestrator,
    water_mask,
)


@dataclass
class CellResult:
    """Outcome of running the full pipeline on one cell."""
    cell_id: str
    duration_s: float = 0.0
    skipped: bool = False
    succeeded: bool = False
    failed_at: str | None = None         # name of the step that raised
    error: str | None = None             # exception class + message
    captured_log: str = ""               # subprocess stdout/stderr (verbose=False)

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        if self.succeeded:
            return "OK"
        return f"FAIL@{self.failed_at or 'unknown'}"


# ---------------------------------------------------------------------------
# Predicates over a Cell's manifest entry — "do we already have this?"
# ---------------------------------------------------------------------------


def _has_water_mask(cell: Cell) -> bool:
    p = cell.water_mask
    return p is not None and p.exists()


def _has_habitat_masks(cell: Cell) -> bool:
    s = cell.seagrass_mask
    o = cell.oyster_mask
    return (s is not None and s.exists()
            and o is not None and o.exists())


def _has_anchors(cell: Cell) -> bool:
    p = cell.anchors_json
    return p is not None and p.exists()


# ---------------------------------------------------------------------------
# Per-cell runner — explicit form (caller manages Area + geojson cache)
# ---------------------------------------------------------------------------


def run_cell(area_id: str,
             cell: Cell,
             oyster_geojson: str,
             seagrass_geojson: str,
             skip_existing: bool = False,
             verbose: bool = False) -> CellResult:
    """Run the full pipeline on one cell. Catches exceptions; never raises.

    If ``skip_existing`` is True and the cell already has an anchors JSON,
    the cell is marked SKIP and no work is done. (The manifest is consulted
    for the existence check; stale-vs-fresh is NOT considered.)

    Captures subprocess stdout/stderr by default so the parent run loop's
    output stays scannable. Pass ``verbose=True`` to see everything as it
    streams (useful when debugging a single cell).
    """
    result = CellResult(cell_id=cell.cell_id)
    start = time.time()

    if skip_existing and _has_anchors(cell):
        result.skipped = True
        result.duration_s = time.time() - start
        return result

    buf = io.StringIO()
    target_out = sys.stdout if verbose else buf
    target_err = sys.stderr if verbose else buf

    try:
        with redirect_stdout(target_out), redirect_stderr(target_err):
            # ---- ensure inputs ----
            if not _has_water_mask(cell):
                result.failed_at = "ensure_water_mask"
                water_mask.run_one(area_id, cell.cell_id)

            if not _has_habitat_masks(cell):
                result.failed_at = "ensure_habitat_masks"
                habitat_mask.process_cell(
                    area_id, cell.cell_id, oyster_geojson, seagrass_geojson,
                )

            # ---- detectors ----
            for name, mod in (
                ("detect_drains",  detect_drains),
                ("detect_islands", detect_islands),
                ("detect_points",  detect_points),
                ("detect_pockets", detect_pockets),
            ):
                result.failed_at = name
                rc = mod.run_one(area_id, cell.cell_id)
                if rc != 0:
                    raise RuntimeError(f"{name}.run_one returned rc={rc}")

            # ---- orchestrator ----
            result.failed_at = "orchestrator"
            rc = orchestrator.run_one(area_id, cell.cell_id)
            if rc != 0:
                raise RuntimeError(f"orchestrator.run_one returned rc={rc}")

            # If we got here, every step passed
            result.succeeded = True
            result.failed_at = None

    except SystemExit as e:
        # SystemExit from inside a step (e.g. unknown cell) — capture without
        # re-raising so the rest of the area still runs.
        result.error = f"SystemExit: {e.code if hasattr(e, 'code') else e}"
    except Exception as e:  # noqa: BLE001 — runner must continue past any failure
        result.error = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()

    if not verbose:
        result.captured_log = buf.getvalue()

    result.duration_s = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper — single-cell entry for cell_analyzer and other callers
# ---------------------------------------------------------------------------


def ensure_habitat_geojsons(area_id: str) -> tuple[str, str]:
    """Ensure FWC habitat geojsons are cached for the area.

    Returns ``(oyster_geojson_path, seagrass_geojson_path)``. These are
    re-used across cells in the area; ``run_cell`` takes them as
    parameters so callers can fetch once per area.
    """
    return (
        habitat_mask.ensure_oyster_geojson(area_id),
        habitat_mask.ensure_seagrass_geojson(area_id),
    )


def run_cell_full(area_id: str, cell_id: str,
                  *, skip_existing: bool = False,
                  verbose: bool = False) -> CellResult:
    """Run the full CV pipeline on one cell, by area_id + cell_id.

    Convenience entry point for callers that only have IDs (e.g.
    ``cell_analyzer`` swapping in for the deprecated ``run_structure_phase``).
    Resolves the Area, ensures habitat geojsons are cached, then delegates
    to ``run_cell``. Each call to this function re-loads the Area; if
    you're iterating many cells, prefer the explicit form.
    """
    area = Area(area_id)
    cell = area.cell(cell_id)
    oyster_gj, seagrass_gj = ensure_habitat_geojsons(area_id)
    return run_cell(
        area_id, cell, oyster_gj, seagrass_gj,
        skip_existing=skip_existing, verbose=verbose,
    )
