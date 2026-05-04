# Deprecated code

This file tracks code that's been superseded by newer architecture but is
kept around until all consumers have migrated. Anything listed here will be
removed in a future cleanup PR; new code should NEVER import from these
modules.

If you need to add a NEW consumer of something here, stop and migrate to
the replacement instead.

---

## Removed

### `src/readwater/pipeline/structure/` — LLM-driven anchor pipeline

Removed in this commit. Pure helpers (`geo`, `grid_overlay`) extracted to
`readwater.pipeline.{geo, grid_overlay}`; LLM agent + extractors + prompts
+ mosaic + seed_validator deleted; `cell_analyzer` now calls
`cv.cell_pipeline.run_cell_full` at the z16 hand-off point. Tests
`tests/test_structure_*.py` and `scripts/smoke_structure_phase.py` were
deleted alongside the implementation. Replacement: the CV pipeline
(`src/readwater/pipeline/cv/`) — deterministic anchor discovery via
`detect_drains` / `detect_islands` / `detect_points` / `detect_pockets`
combined by `orchestrator`. Downstream consumers should read anchors via
`Area(area_id).cell(cell_id).anchors_json`.

---

## `scripts/_cells.py` — hand-picked 9-cell test fixture

**Status:** deprecated for CV pipeline, still imported by 3 non-CV scripts.

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
  - `scripts/prompt_experiment.py`        — prompt evaluation harness

These three scripts pre-date the manifest and will be migrated (or
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
