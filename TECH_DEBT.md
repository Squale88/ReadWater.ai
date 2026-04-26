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

### 2. ~~Dead path: `analyze_structure_image()` and `structure_analysis_*` prompts~~ — RESOLVED (Phase C TASK-7)
Deleted `analyze_structure_image()` from `src/readwater/api/claude_vision.py`, the prompt pair `prompts/structure_analysis_{system,user}.txt`, and the corresponding tests in `tests/test_claude_vision.py`. Replaced by `readwater.pipeline.structure.anchor_discovery.run_anchor_discovery` (see `docs/PHASE_C_DISCOVERY_PIPELINE.md`).

### 3. Duplicate grid-drawing logic
Two independent implementations of "draw a labeled grid on an image":
- [src/readwater/pipeline/image_processing.py](src/readwater/pipeline/image_processing.py) — fixed 4×4, numbered 1–16. Used by the recursive cell analyzer.
- [src/readwater/pipeline/structure/grid_overlay.py](src/readwater/pipeline/structure/grid_overlay.py) — flexible rows×cols with A1-style labels. Used by the structure phase.

Both reimplement font loading, line drawing, and row/col label math. `grid_overlay` is the more general one; consider having `image_processing` delegate to it (or at least share the label helper).

### 4. ~~Ground-truth evidence only feeds the discovery prompt~~ — partially RESOLVED (Phase C TASK-3 + TASK-7)
Phase C v1 deleted the legacy `discover_anchors`, `identify_anchor`, `identify_subzones`, and `resolve_continuation` prompts/code paths. The new path in `anchor_discovery.run_anchor_discovery` injects habitat evidence into the v3 prompts via `evidence.build_cell_evidence_section()` (controlled by `AnchorDiscoveryConfig.inject_evidence`). The evidence-into-coord-gen path is intentionally NOT wired (coord-gen is pure spatial localization, not classification).

Phase D substructure work, when it lands, will need its own evidence wiring decisions.

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

### 7. Phase C TASK-7 leftovers (smaller follow-ups)
The Phase C cleanup pass kept its scope tight. Items still owed:
- `scripts/poc_grid_discovery.py`, `scripts/poc_grid_identify.py`, `scripts/smoke_structure_phase.py` — hardcoded Windows paths (item 1) plus references to deleted prompts. Either repath via `Path(__file__).resolve().parents[1]` or archive under `scripts/archive/`.
- `scripts/prompt_experiment.py` — references `anchor_identification_*_v2.txt` (deleted). Update to reference v3 nogrid/grid pairs or archive.
- `scripts/continue_depth2.py` and `tests/test_pipeline.py` `_summary.json` reads — confirm whether `continue_depth2.py` outputs are still needed; delete if not.
- `prompts/grid_scoring_user2.txt` lives on (still loaded by `claude_vision.py:187`). The naming is misleading — should be `grid_scoring_user_no_lean.txt` or similar. Rename in a separate pass.
- `tests/test_structure_extractors.py` and `tests/test_structure_seed_validator.py` — verify they exercise live code (Phase D extractors should still need them).

## Done
<!-- Move items here with commit hash when fixed. -->
- Phase C TASK-0..TASK-7 (see `docs/PHASE_C_TASKS.md` and `docs/PHASE_C_DISCOVERY_PIPELINE.md`):
  - GT files for 6 cells, AnchorStructure schema (PhaseEvent/Provenance/Finding/Z18FetchPlan), z18 tile plan helper, v3 prompt grid/nogrid split, coord-gen prompt grid/nogrid split, habitat-evidence injection, anchor_discovery module, agent.py rewrite (1113 → 167 lines), 23 dead prompt files removed, dead `analyze_structure_image()` removed, legacy `prompts.py` LLM wrappers deleted, legacy `tests/test_structure_agent.py` deleted, eval harness with split id/placement reports + per-cell PNG overlays.
