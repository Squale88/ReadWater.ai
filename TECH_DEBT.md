# Tech Debt & Follow-ups

Known issues and cleanup items to tackle when it makes sense. Add new items at the bottom; remove items when fixed (reference the commit).

## Open

### 1. Hardcoded Windows paths in every script under `scripts/`
Ten files declare `REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")`. The current checkout lives at `C:\claude-projects\ReadWater.ai`, so none of the scripts run as-is.

- [scripts/_cells.py:32](scripts/_cells.py#L32)
- [scripts/smoke_structure_phase.py:26](scripts/smoke_structure_phase.py#L26) (also `OUT_ROOT` at [line 46](scripts/smoke_structure_phase.py#L46))
- [scripts/prompt_experiment.py:48](scripts/prompt_experiment.py#L48)
- [scripts/poc_grid_discovery.py:30](scripts/poc_grid_discovery.py#L30)
- [scripts/poc_grid_identify.py:32](scripts/poc_grid_identify.py#L32)
- [scripts/poc_cv_segmentation.py:25](scripts/poc_cv_segmentation.py#L25)
- [scripts/fwc_habitat_mask.py:27](scripts/fwc_habitat_mask.py#L27)
- [scripts/naip_water_mask.py:26](scripts/naip_water_mask.py#L26)
- [scripts/noaa_channel_mask.py:28](scripts/noaa_channel_mask.py#L28)

**Fix:** Replace with `REPO_ROOT = Path(__file__).resolve().parents[1]` (or an env var). One sweep, no behavior change.

### 2. Dead path: `analyze_structure_image()` and `structure_analysis_*` prompts
[src/readwater/api/claude_vision.py:285](src/readwater/api/claude_vision.py#L285) loads `prompts/structure_analysis_system.txt` and `prompts/structure_analysis_user.txt`. Neither is called by `cell_analyzer.py` — the recursive pipeline now hands z16 cells to `run_structure_phase` in [src/readwater/pipeline/structure/](src/readwater/pipeline/structure/) instead. The function still has unit tests in [tests/test_claude_vision.py](tests/test_claude_vision.py).

**Decide:** delete entirely, or keep as a documented fallback. Either way, drop the tests or retarget them.

### 3. Duplicate grid-drawing logic
Two independent implementations of "draw a labeled grid on an image":
- [src/readwater/pipeline/image_processing.py](src/readwater/pipeline/image_processing.py) — fixed 4×4, numbered 1–16. Used by the recursive cell analyzer.
- [src/readwater/pipeline/structure/grid_overlay.py](src/readwater/pipeline/structure/grid_overlay.py) — flexible rows×cols with A1-style labels. Used by the structure phase.

Both reimplement font loading, line drawing, and row/col label math. `grid_overlay` is the more general one; consider having `image_processing` delegate to it (or at least share the label helper).

### 4. Ground-truth evidence only feeds the discovery prompt
[src/readwater/pipeline/evidence.py](src/readwater/pipeline/evidence.py) builds water/channel/oyster/seagrass coverage tables and injects them into `discover_anchors`. The downstream prompts (`identify_anchor`, `identify_subzones`, `resolve_continuation`) receive no evidence — an oyster-bar identify call doesn't see the oyster-survey mask.

**Fix:** extend evidence injection into the identify/resolve prompts, scoped to layers relevant to the feature type.

### 5. Test coverage gaps
- No end-to-end integration test that exercises `analyze_cell()` recursion → structure phase together (only mocked unit tests + the live `smoke_structure_phase.py`).
- `render_annotated()` in [src/readwater/pipeline/structure/mosaic.py](src/readwater/pipeline/structure/mosaic.py) has no test coverage.
- Overlap conflict resolution policy (OverlapEntry kept/subordinated) is recorded but not tested.

### 6. Pre-existing failures in `tests/test_pipeline.py`
Four tests fail on `master` before any Phase-1 retained-cell-context work:
- `test_hard_cap_stops_at_limit`
- `test_hard_cap_tree_structure`
- `test_hard_cap_metadata`
- `test_structure_api_calls_count_per_provider`

All four assert specific `state.api_calls` / tree counts after a hard-cap run. They appear to be stale after earlier pipeline refactors that changed what counts as an "API call." Not caused by the retained-cell-context branch — verified by running the same four against `master` independently. Fix is a rewrite of the assertions against the current accounting model, not the production code.

## Done
<!-- Move items here with commit hash when fixed. -->
