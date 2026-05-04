# Deprecated code

This file tracks code that's been superseded by newer architecture but is
kept around until all consumers have migrated. Anything listed here will be
removed in a future cleanup PR; new code should NEVER import from these
modules.

If you need to add a NEW consumer of something here, stop and migrate to
the replacement instead.

---

## `src/readwater/pipeline/structure/` — LLM-driven anchor pipeline

**Status:** deprecated, but still imported by `cell_analyzer.py`. A future
PR will rewire `cell_analyzer` to stop calling `run_structure_phase`; once
that lands, this whole subtree can be deleted.

**Replaced by:** `src/readwater/pipeline/cv/` (deterministic CV-based
anchor discovery). The CV pipeline produces equivalent anchor records
without per-cell Claude vision calls — see
`src/readwater/pipeline/cv/orchestrator.py` and the surrounding detector
modules.

**Files:**

  - `structure/__init__.py`            — re-exports `run_structure_phase`
  - `structure/agent.py`               — state-machine orchestrator (Claude vision)
  - `structure/prompts.py`             — LLM prompt templates
  - `structure/seed_validator.py`      — anchor-seed verification step
  - `structure/mosaic.py`              — z18 mosaic helpers
  - `structure/grid_overlay.py`        — grid-cell rendering
  - `structure/geo.py`                 — geo helpers
  - `structure/extractors/`            — region/clickbox/gridcell extractors

**Known live consumers (these import the deprecated code today):**

  - `src/readwater/pipeline/cell_analyzer.py` — calls `run_structure_phase`
    at z16 during recursive discovery. Future rewire: skip the call (CV
    runs separately via `scripts/run_area.py`) or delegate to it.
  - `tests/test_structure_*.py` — test files for the deprecated agent.
    Will be deleted with the implementation.
  - `scripts/smoke_structure_phase.py`, `scripts/prompt_experiment.py` —
    legacy validation scripts.

**Migration path for new code:**

  1. Discovery still happens via `cell_analyzer.py` (until the rewire).
  2. CV pipeline runs via `python scripts/run_area.py --area <id>` and
     produces `cv_all_*.json` per cell.
  3. Downstream consumers (trip planner, future LLM interpretation phase)
     should read the anchor JSON via `Area("...").cell(cid).anchors_json`
     — not from any structure-pipeline output.

---

## `scripts/_cells.py` — hand-picked 9-cell test fixture

**Status:** deprecated for CV pipeline, still imported by 4 non-CV scripts.

**Replaced by:** `readwater.areas.Area`. Cells, centers, and parent bboxes
are now indexed in `data/areas/<area>/manifest.json` and exposed through
typed accessors:

```python
from readwater.areas import Area

area = Area("rookery_bay_v2")
for cell in area.cells():               # all 100 cells, not just the 9 hand-picked
    print(cell.cell_id, cell.center, cell.water_mask)
```

**Known live consumers:**

  - `scripts/noaa_channel_mask.py`        — NOAA channel mask fetcher
  - `scripts/fetch_naip_tifs.py`          — NAIP raster fetcher
  - `scripts/smoke_structure_phase.py`    — legacy structure-pipeline test
  - `scripts/prompt_experiment.py`        — prompt evaluation harness

These four scripts pre-date the manifest and will be migrated (or
deprecated themselves) in follow-up PRs.

---

## NOT deprecated, despite the name suggesting otherwise

### `src/readwater/pipeline/water_mask.py` — NDWI primitives

This module looks like it should be deprecated alongside the NDWI-only
water-mask approach (which IS sunset — the new pipeline uses Google styled
tiles + NAIP carve + connectivity filtering, not NDWI alone). But the
module itself only contains shared math primitives:

  - `compute_ndwi(green, nir)`
  - `threshold_water(ndwi, threshold)`
  - `load_4band_tif(...)`
  - `save_mask_png(...)`, `save_mask_overlay_png(...)`, etc.

These primitives are imported by the NEW
`src/readwater/pipeline/cv/water_mask.py` to build the NAIP carve pass.
Don't deprecate this module. If we ever rename it for clarity (e.g. to
`ndwi.py`), do so as a refactor with full call-site updates.
