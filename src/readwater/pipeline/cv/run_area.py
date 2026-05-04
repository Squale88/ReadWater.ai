"""End-to-end CV pipeline runner for an entire area.

For each cell in the requested area, runs the full pipeline in sequence:

  1. water_mask.run_one      (Google styled tile + NAIP carve + connectivity)
  2. habitat_mask.process_cell (FWC seagrass + oyster polygon rasterization)
  3. detect_drains.run_one
  4. detect_islands.run_one
  5. detect_points.run_one
  6. detect_pockets.run_one
  7. orchestrator.run_one    (dedup + cluster + parent/child links)

After all cells complete, regenerates ``manifest.json`` so the index
reflects the new outputs.

Usage:
  python scripts/run_area.py --area rookery_bay_v2
  python scripts/run_area.py --area rookery_bay_v2 --cell root-10-8 --cell root-11-5
  python scripts/run_area.py --area rookery_bay_v2 --skip-existing
  python scripts/run_area.py --area rookery_bay_v2 --verbose

Defaults (per the migration's design call):
  - Idempotency: re-runs every cell. Pass ``--skip-existing`` to skip
    cells that already have an anchors JSON.
  - Errors: log and continue. The summary at the end lists every failed
    cell with the failing step and the captured tail of subprocess output.
  - Verbosity: per-cell subprocess output is captured silently; only the
    one-line per-cell status is shown. Pass ``--verbose`` to see
    everything as it streams.

Output layout (per cell, all under ``data/areas/<area>/``):

  masks/water/<cell>_water_mask.png     (and styled debug tiles)
  masks/seagrass/<cell>_seagrass_mask.png
  masks/oyster/<cell>_oyster_mask.png
  images/structures/<cell>/cv_drains_<ts>.{png,json}
  images/structures/<cell>/cv_islands_<ts>.{png,json}
  images/structures/<cell>/cv_points_<ts>.{png,json}
  images/structures/<cell>/cv_pockets_<ts>.{png,json}
  images/structures/<cell>/cv_all_<ts>.{png,json}        (anchors + overlay)
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path

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

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUILD_MANIFEST_SCRIPT = _REPO_ROOT / "scripts" / "build_manifest.py"


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
# Per-cell runner
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
# CLI
# ---------------------------------------------------------------------------


def _resolve_cells(area: Area, requested: list[str]) -> list[Cell]:
    if not requested:
        return list(area.cells())
    bad = [c for c in requested if not area.has_cell(c)]
    if bad:
        raise SystemExit(
            f"unknown cell(s) for area {area.area_id!r}: {bad!r}"
        )
    return [area.cell(cid) for cid in requested]


def _print_failures(results: list[CellResult], log_tail_lines: int = 20) -> None:
    print()
    print("Failures:")
    for r in results:
        if r.succeeded or r.skipped:
            continue
        print(f"  {r.cell_id:<14s} {r.failed_at or '-':<22s} {r.error or ''}")
        if r.captured_log:
            tail = r.captured_log.splitlines()[-log_tail_lines:]
            for line in tail:
                print(f"    | {line}")


def _rebuild_manifest(area_id: str) -> int:
    """Re-run scripts/build_manifest.py to refresh the area's manifest.json.

    Subprocess invocation is deliberate: the build_manifest entry point is
    a script today and we don't want to import its CLI plumbing here. It's
    cheap (~tens of ms) and only runs once per area run.
    """
    rc = subprocess.call(
        [sys.executable, str(_BUILD_MANIFEST_SCRIPT), "--area", area_id],
    )
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--area", default="rookery_bay_v2",
        help="Area id (default: rookery_bay_v2).",
    )
    parser.add_argument(
        "--cell", action="append", default=[],
        help="Limit run to specific cells. Repeatable. Default: every cell "
             "in the area's manifest.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip cells that already have an anchors JSON in the manifest. "
             "Default: re-run every cell (per the design call for explicit "
             "idempotency).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream per-cell subprocess output as it happens. Default: "
             "capture silently and only show the captured tail on failure.",
    )
    parser.add_argument(
        "--no-rebuild-manifest", action="store_true",
        help="Skip the post-run manifest rebuild (useful in tight inner loops).",
    )
    args = parser.parse_args()

    area = Area(args.area)
    cells = _resolve_cells(area, args.cell)

    print(f"Running CV pipeline on {len(cells)} cell(s) in area {args.area!r}")
    print(f"  output root: {area.path}")
    if args.skip_existing:
        print(f"  --skip-existing: cells with existing anchors JSON will be skipped")
    if args.verbose:
        print(f"  --verbose: per-cell subprocess output will stream live")
    print()

    # Habitat geojsons are area-level cached resources; resolve once before
    # the cell loop so process_cell() doesn't re-fetch per cell.
    print("Ensuring habitat geojsons cached...")
    oyster_gj = habitat_mask.ensure_oyster_geojson(args.area)
    seagrass_gj = habitat_mask.ensure_seagrass_geojson(args.area)
    print()

    run_started = time.time()
    results: list[CellResult] = []
    for i, cell in enumerate(cells, start=1):
        result = run_cell(
            args.area, cell, oyster_gj, seagrass_gj,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
        results.append(result)
        msg = (f"  [{i:>3}/{len(cells)}] {cell.cell_id:<14s} "
               f"{result.status:>14s}  ({result.duration_s:>5.1f}s)")
        if result.error:
            tail = result.error[:80].replace("\n", " ")
            msg += f"  {tail}"
        print(msg, flush=True)

    total_time = time.time() - run_started
    n_ok = sum(1 for r in results if r.succeeded)
    n_skip = sum(1 for r in results if r.skipped)
    n_fail = sum(1 for r in results if not r.succeeded and not r.skipped)

    print()
    print("=" * 70)
    print(f"Summary: {n_ok} OK, {n_skip} SKIP, {n_fail} FAIL  "
          f"(total {total_time:.0f}s, avg {total_time / max(1, len(cells)):.1f}s/cell)")

    if n_fail:
        _print_failures(results)

    if not args.no_rebuild_manifest:
        print()
        print("Rebuilding manifest...")
        rc = _rebuild_manifest(args.area)
        if rc != 0:
            print(f"  WARN: build_manifest exited rc={rc}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
