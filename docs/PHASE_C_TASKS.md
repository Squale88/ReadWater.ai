# Phase C Task Cards

Work items to complete Phase C (Anchor Identification) per `docs/PIPELINE_PHASES.md`.

**Locked decisions (from design discussion):**
- Adopt v3 anchor identification (retire current z16 grid-cell DISCOVER).
- Ship **two v3 variants**: `_nogrid` and `_grid`, toggleable in the tuning harness.
- Coord-gen is a **separate second call** that emits **pixel coordinates**; Python converts to lat/lon via `structure/geo.py:pixel_to_latlon`.
- Coord-gen **receives the grid-overlaid z16 image** as a reference frame.
- **One** coord-gen prompt pair (works for both v3 variants).
- Coord-gen batches all anchors in a single call to start.
- Z18-mosaic grid-cell IDENTIFY / EXTRACT for Phase D substructures **stays** — unrelated to this refactor.

**Dependency graph:**

```
TASK-0 (GT upload)           TASK-4 (schema)     TASK-5 (tile helper)
       │                           │                     │
       ▼                           │                     │
TASK-1 (v3 variants)               │                     │
       │                           │                     │
       ├─► TASK-2 (coord-gen) ─────┤                     │
       │                           │                     │
       ▼                           │                     │
TASK-3 (habitats)                  │                     │
       │                           │                     │
       └───────────────┬───────────┴─────────────────────┘
                       ▼
              TASK-6 (wire into agent)
                       │
                       ▼
              TASK-7 (cleanup)
```

Effort legend: **S** = under 2 hrs, **M** = half day, **L** = multi-day.

---

## TASK-0 — Ingest ground-truth image and anchor file

**Goal.** Land the hand-labeled GT for root-10-8 (test image + anchor file you already prepared) into the repo so the coord-gen tuning harness has something to measure against.

**Depends on.** User upload.
**Blocks.** TASK-2.
**Effort.** S.

**Files touched.**
- `data/areas/rookery_bay_v2/images/structures/root-10-8/gt_anchors.json` (new)
- `data/areas/rookery_bay_v2/images/structures/root-10-8/gt_overlay.png` (new — your labeled image)

**Steps.**
1. Drop the image and anchor file into `data/areas/rookery_bay_v2/images/structures/root-10-8/`.
2. Normalize the anchor file to this shape (one entry per GT anchor; matches the existing `GTAnchor` dataclass style in `tune_anchor_identification_v3.py`):
   ```json
   {
     "cell_id": "root-10-8",
     "z16_image": "data/areas/rookery_bay_v2/images/z0_10_8.png",
     "z16_center": [lat, lon],
     "image_size_px": [1280, 1280],
     "anchors": [
       {
         "gt_id": "gt1",
         "label": "NW drain system",
         "structure_type_options": ["drain_system", "creek_mouth_system"],
         "pixel_center": [x, y],
         "pixel_bbox": [x0, y0, x1, y1],
         "latlon_center": [lat, lon]
       }
     ]
   }
   ```
3. Pixel centers round-trip through `geo.pixel_to_latlon` — verify the `latlon_center` matches before committing.

**Acceptance criteria.**
- GT file lives at the path above and parses as JSON.
- Running a quick `pixel_to_latlon` check on one anchor matches its recorded `latlon_center` within <5 m.

---

## TASK-1 — Split v3 into `_nogrid` and `_grid` variants + harness toggle

**Goal.** Fork the current v3 prompt into two variants (with and without grid overlay) and wire a `--grid-mode {none, grid, both}` flag into the tuning harness so you can A/B cleanly on the same cell.

**Depends on.** None.
**Blocks.** TASK-2, TASK-3, TASK-6.
**Effort.** M.

**Files touched.**
- `prompts/anchor_identification_v3_nogrid_system.txt` (rename of current `anchor_identification_system_v3.txt`)
- `prompts/anchor_identification_v3_nogrid_user.txt` (rename of current `anchor_identification_user_v3.txt`)
- `prompts/anchor_identification_v3_grid_system.txt` (new; copy of nogrid + diffs below)
- `prompts/anchor_identification_v3_grid_user.txt` (new)
- `scripts/tune_anchor_identification_v3.py`

**Steps.**
1. Rename current v3 files with the `_nogrid` suffix.
2. Create `_grid` variants, diffs from `_nogrid`:
   - Line 11: change "No grid overlay is used. All positions are described relative to zones and image quadrants." → "A grid overlay IS present on image 1 (A1…H8, 8×8). Use it as a precise spatial reference. Every anchor MUST also include compass positions — do not drop the compass language."
   - Section "Position language requirement" (lines 49–53): add "Each anchor must also include a `grid_cells` field listing the grid-cell labels the anchor occupies."
   - Output schema (line 272 onward): add `"grid_cells": ["A1", "B2"]` field to each anchor.
3. Add `--grid-mode {none, grid, both}` to `tune_anchor_identification_v3.py`:
   - `none` → load `_nogrid` pair, send z16 image as-is.
   - `grid` → load `_grid` pair, render grid overlay on z16 image using `structure/grid_overlay.py`, send overlaid image.
   - `both` → run both modes sequentially, emit a side-by-side report (GT hits per mode, anchor count per mode, agreement set).
4. Cache overlaid image under `data/areas/rookery_bay_v2/images/z0_10_8_grid.png` so reruns don't redraw.

**Acceptance criteria.**
- `python scripts/tune_anchor_identification_v3.py --grid-mode none` passes current GT scoring (no regression).
- `python scripts/tune_anchor_identification_v3.py --grid-mode grid` runs end-to-end, outputs anchors with `grid_cells` populated.
- `python scripts/tune_anchor_identification_v3.py --grid-mode both` prints side-by-side report: GT-match counts, anchor counts, which anchors each mode uniquely surfaced.

---

## TASK-2 — Build coord-gen prompt pair + tuning harness

**Goal.** Given a v3 anchor list and the grid-overlaid z16 image, produce pixel centerpoint + pixel bbox per anchor. Convert pixels → lat/lon via `geo.pixel_to_latlon`. Measure per-anchor distance error in meters against GT.

**Depends on.** TASK-0, TASK-1.
**Blocks.** TASK-6.
**Effort.** M.

**Files touched.**
- `prompts/anchor_coords_system.txt` (new)
- `prompts/anchor_coords_user.txt` (new)
- `scripts/tune_anchor_coords.py` (new)

**Steps.**
1. Write `anchor_coords_system.txt`:
   - Role: "Given a z16 satellite image (1280×1280 px, with 8×8 grid overlay A1…H8) and a list of anchors already identified on it with compass positions, structure types, size fractions, zone assignments, and sub-features, output a pixel centerpoint and pixel bounding box for each anchor on the image."
   - Rules: "Do not invent new anchors. Do not merge, rename, or drop. One output entry per input anchor, same `anchor_id`. If you cannot confidently locate an anchor, still emit a best-effort center and set `placement_confidence` low."
   - Rule on bbox: "`pixel_bbox` must be the tightest axis-aligned rectangle that contains the anchor's visible footprint on the z16 image. For linear troughs, the bbox is long and thin along the shoreline."
2. Write `anchor_coords_user.txt`: takes `{cell_id}`, `{z16_center_lat}`, `{z16_center_lon}`, `{anchor_list_json}` (from the v3 run's `parsed.json`).
3. Output schema (batched):
   ```json
   {
     "anchors": [
       {
         "anchor_id": "a1",
         "pixel_center": [x, y],
         "pixel_bbox": [x0, y0, x1, y1],
         "placement_confidence": 0.0,
         "placement_notes": "..."
       }
     ]
   }
   ```
4. Write `scripts/tune_anchor_coords.py`:
   - Arg: `--v3-run <path to tuning_runs/v3_YYYYMMDD-HHMMSS/>` (defaults to latest).
   - Loads `parsed.json` from that run → anchor list.
   - Loads grid-overlaid z16 image.
   - Calls coord-gen, extracts JSON.
   - Converts each `pixel_center` → `(lat, lon)` via `pixel_to_latlon`.
   - Matches to GT anchors (reuse the greedy match logic from `tune_anchor_identification_v3.py` by `anchor_id` if possible, otherwise by structure_type + position overlap).
   - Reports per-anchor **meters error** (haversine between predicted and GT latlon_center) + bbox IoU vs. GT pixel_bbox.
   - Saves run artifacts to `data/areas/rookery_bay_v2/images/coord_runs/YYYYMMDD-HHMMSS/`.
5. Add `--replay` mode same as v3 harness.

**Acceptance criteria.**
- Single call returns a placement for every v3 anchor (no dropped anchors).
- Per-anchor meters error is reported.
- Baseline run on the existing v3 root-10-8 output produces a report; absolute accuracy is a later tuning concern, not acceptance.

---

## TASK-3 — Inject habitat evidence into v3 prompts

**Goal.** Feed NOAA channel, FWC oyster/seagrass, and NAIP water/land evidence masks into both v3 variants as supporting context, so anchor classification (especially `oyster_bar`) can cross-reference surveyed data.

**Depends on.** TASK-1.
**Blocks.** TASK-6.
**Effort.** S–M.

**Files touched.**
- `prompts/anchor_identification_v3_nogrid_system.txt`
- `prompts/anchor_identification_v3_grid_system.txt`
- `prompts/anchor_identification_v3_nogrid_user.txt`
- `prompts/anchor_identification_v3_grid_user.txt`
- `scripts/tune_anchor_identification_v3.py`

**Steps.**
1. Import `evidence.build_evidence_table()` into the tuning harness; call it using the context bundle's overlay summaries.
2. Add `{evidence_table}` placeholder to both `_user.txt` templates (after the context_bundle JSON section).
3. Add a short block to both `_system.txt` prompts, after "Context package handling" (line 22):
   > A habitat evidence table may be provided below (FWC surveyed oyster reefs, NOAA-charted channels, NAIP water/land masks). Treat it as SUPPORTING EVIDENCE, not ground truth. An anchor's oyster_bar classification is more credible when the evidence table confirms oyster substrate in that region; a drain_system is more credible when NOAA shows a charted channel nearby. Never override image 1 — if the image contradicts the evidence table, trust the image and note the contradiction.
4. Re-run GT scoring on root-10-8 with and without evidence to confirm no regression.

**Acceptance criteria.**
- Harness fills evidence table into prompts end-to-end.
- GT match count on root-10-8 is at least as good as pre-evidence (ideally better for oyster-type anchors).
- Anchor rationale fields reference evidence entries when used.

---

## TASK-4 — Extend `AnchorStructure` schema

**Goal.** Bring `AnchorStructure` up to the v1 Phase C contract in `docs/PIPELINE_PHASES.md` — add state, phase_history, provenance, findings, seed Z18FetchPlan, priority rank, and zone linkage.

**Depends on.** None.
**Blocks.** TASK-6.
**Effort.** M.

**Files touched.**
- `src/readwater/models/structure.py`
- `src/readwater/pipeline/structure/agent.py` (populate new fields where anchors are emitted)
- `tests/test_pipeline.py` (update fixtures)

**Steps.**
1. New Pydantic types in `models/structure.py`:
   - `PhaseEvent { phase: str, action: str, actor: str, timestamp: str, note: str | None }`
   - `Provenance { source_images: list[str], overlay_refs: list[str], prompt_id: str, prompt_version: str, provider_config: dict, input_hash: str }`
   - `Finding { issue_code: str, severity: Literal["info","warn","error"], object_id: str, field: str | None, message: str, recommended_action: str | None }`
   - `Z18FetchPlan { tile_centers: list[tuple[float, float]], tile_budget: int, extent_meters: float }`
2. Add to `AnchorStructure`:
   - `state: Literal["draft", "validated", "approved", "rejected"] = "draft"`
   - `phase_history: list[PhaseEvent] = []`
   - `provenance: Provenance`
   - `findings: list[Finding] = []`
   - `seed_z18_fetch_plan: Z18FetchPlan | None = None`
   - `priority_rank: int | None = None`
   - `zone_id: str | None = None` (v3 emits zones; links to `LocalComplex`)
3. Populate fields in `agent.py` where `AnchorStructure(...)` is constructed today (line ~835). `state = "draft"` on emit; `phase_history` appended with the DISCOVER event.
4. Update any fixtures/tests that instantiate `AnchorStructure` directly.

**Acceptance criteria.**
- `pytest tests/` passes.
- New anchors emitted by `run_structure_phase()` carry `state="draft"`, a non-empty `phase_history`, and fully-populated `provenance`.

---

## TASK-5 — lat/lon → z18 tile plan helper in `mosaic.py`

**Goal.** Replace the grid-cell-based tile-plan computation (which assumed z16 cell labels) with a lat/lon-native helper that takes an anchor center + rough extent and emits a z18 `Z18FetchPlan`.

**Depends on.** TASK-4 (uses the `Z18FetchPlan` type).
**Blocks.** TASK-6.
**Effort.** S.

**Files touched.**
- `src/readwater/pipeline/structure/mosaic.py`
- `tests/test_mosaic.py` (new or extended)

**Steps.**
1. Add function:
   ```python
   def z18_tile_plan_from_latlon(
       anchor_center_latlon: tuple[float, float],
       rough_extent_meters: float,
       tile_budget: int = 25,
   ) -> Z18FetchPlan
   ```
   Math: at z18, one Google Static tile (640×640 × scale=2 = 1280 px) covers ~150 m per side. Walk outward from the anchor center in both axes until the coverage exceeds `rough_extent_meters * 1.25` (25% padding), capped by `tile_budget`.
2. Return the list of tile centers (lat, lon) in row-major order, plus the effective extent.
3. Unit tests: known anchor center → expected number of tiles, first and last tile centers within tolerance.

**Acceptance criteria.**
- Unit tests pass.
- Given a 200 m extent, emits a 3×3 tile grid centered on the anchor.
- Honors `tile_budget` (never exceeds).

---

## TASK-6 — Wire v3 + coord-gen into `run_structure_phase()`

**Goal.** Replace the current z16 grid-cell DISCOVER / IDENTIFY with: v3 DISCOVER → COORDS → PLAN_CAPTURE (using TASK-5 helper). Produce `AnchorStructure` objects with populated lat/lon centers and seed fetch plans. Z18-mosaic decomposition for substructures (Phase D work) remains unchanged.

**Depends on.** TASK-1, TASK-2, TASK-3, TASK-4, TASK-5.
**Blocks.** TASK-7.
**Effort.** L.

**Files touched.**
- `src/readwater/pipeline/structure/agent.py`
- `src/readwater/pipeline/structure/prompts.py` (add v3 + coord-gen wrappers)
- `src/readwater/pipeline/structure/__init__.py` (export new helpers if needed)
- `tests/test_pipeline.py` (integration test)

**Steps.**
1. Add `discover_anchors_v3()` to `prompts.py`:
   - Takes the `Z16ContextBundle`, grid-mode config (`"nogrid" | "grid"`), evidence table.
   - Loads the correct prompt pair from `prompts/anchor_identification_v3_{mode}_{system,user}.txt`.
   - Sends the 4-image handoff (+ grid-overlaid z16 when in grid mode).
   - Returns parsed JSON with zones + anchors.
2. Add `locate_anchor_coords()` to `prompts.py`:
   - Takes v3 anchor list + grid-overlaid z16 image.
   - Calls coord-gen prompt.
   - Returns `{anchor_id: {pixel_center, pixel_bbox, placement_confidence}}`.
3. In `agent.py`:
   - Replace the `DISCOVER` stage (~lines 103–193) with a call to `discover_anchors_v3()`.
   - Insert a new `COORDS` stage between DISCOVER and RANK_AND_DEFER.
   - In COORDS: call `locate_anchor_coords()`, then for each anchor convert `pixel_center` → `anchor_center_latlon` via `geo.pixel_to_latlon`, and build `geometry.latlon_polygon` from `pixel_bbox` corners.
   - Populate `AnchorStructure.zone_id` from v3's `zone_id` field.
   - In PLAN_CAPTURE: call `mosaic.z18_tile_plan_from_latlon()` instead of the old cell-based path; store result as `anchor.seed_z18_fetch_plan`.
   - Append `PhaseEvent`s to `phase_history` at each stage transition.
4. Delete the old `discover_anchors()` function that wraps the grid-cell prompt, along with its helpers that are no longer referenced.
5. Integration test in `tests/test_pipeline.py`: run `run_structure_phase()` on a cached root-10-8 fixture, assert at least 3 anchors with `anchor_center_latlon` populated and `seed_z18_fetch_plan.tile_centers` non-empty.

**Acceptance criteria.**
- `run_structure_phase()` on root-10-8 emits `AnchorStructure[]` with non-null lat/lon centers.
- Each anchor carries a `seed_z18_fetch_plan` that covers its rough extent.
- Old grid-cell z16 DISCOVER code path is deleted (not just disabled).
- Integration test passes.

---

## TASK-7 — Cleanup dead prompts and scripts

**Goal.** Remove orphaned prompts, scripts, and code paths left over from earlier iterations. Do this only after TASK-6 is green.

**Depends on.** TASK-6.
**Blocks.** Nothing.
**Effort.** S–M.

**Files touched.** (deletions / archives)

**Steps.**
1. Delete `prompts/anchor_identification_system_v2.txt` and `anchor_identification_user_v2.txt` (v3 supersedes).
2. Consolidate `prompts/grid_scoring_user*.txt` variants — keep one canonical `grid_scoring_user.txt`, delete the `_old`, `_last_working`, `2` variants.
3. Investigate `prompts/working_v1/` — archive under `old_prompts/` or delete.
4. Confirm `analyze_structure_image()` in `src/readwater/api/claude_vision.py:285` is dead (no callers in `src/`). If dead, delete it along with `prompts/structure_analysis_{system,user}.txt`.
5. Confirm nothing reads `continue_depth2.py`'s outputs (`depth2_checkpoint.json`, `depth2_summary.json`). `run_retained_context_test.py` reads `depth2_summary.json` — so keep `depth2_summary.json`, but the script itself can go if its output is already on disk and won't need regenerating.
6. Hardcoded-Windows-path scripts (`scripts/poc_grid_discovery.py`, `scripts/poc_grid_identify.py`, `scripts/smoke_structure_phase.py`):
   - Fix the `REPO_ROOT` bootstrap to use `Path(__file__).resolve().parents[1]`, OR
   - Archive under `scripts/archive/` with a README noting they target the pre-v3 grid-cell path.
7. Update `TECH_DEBT.md` — strike items resolved by this cleanup.

**Acceptance criteria.**
- No unused prompt files remain in `prompts/`.
- `pytest` still passes.
- `TECH_DEBT.md` reflects current state.

---

## Addendum — locked policies (post-design review)

These decisions resolve open questions surfaced during review of the cards above. **Where they conflict with task text, the addendum wins.**

### TASK-2 — anchor matching is strict

- Match coord-gen output to v3 input by `anchor_id` only. The "otherwise by structure_type + position overlap" fallback in TASK-2 step 4 is removed.
- When coord-gen returns an `anchor_id` not in the v3 input list (or fails to return one for an input anchor), the harness logs a warning and increments an `unmatched_anchors` counter in the report. No fuzzy attribution — fuzzy matching would hide the prompt-quality regressions the harness is meant to surface.

### TASK-2 / TASK-6 — coord-gen failure policy

- **Per-anchor failures (out-of-bounds pixel, `placement_confidence` near zero):** keep the anchor, set `seed_z18_fetch_plan=None`, attach a `Finding` (severity=`warn`) describing the failure. PLAN_CAPTURE in TASK-6 skips anchors without coords. Phase E surfaces the findings to the user.
- **Whole-batch JSON malformed / unparseable:** fail the COORDS stage (no recovery possible).
- No `raise` paths for individual bad placements — aligns with v1's "pure findings, state machine carries the truth" stance.

### TASK-4 — provenance backfill, not regenerate

- New `AnchorStructure` instances carry real `Provenance`.
- Existing cached anchor JSON (tuning runs, fixtures) gets a one-shot migration that backfills `Provenance` with `prompt_version="legacy_pre_v1"`, `prompt_id="unknown"`, empty `provider_config`. Honest about being legacy; future queries can filter on `prompt_version` prefix when provenance accuracy matters.

### TASK-0 — GT entries carry a `status` field

Add `status: Literal["active", "under_review", "excluded"]` to each GT anchor. Match-scoring includes only `status="active"`. Initial assignment for `root-10-8`:

| gt_id | Label                                         | Status         | Notes                                                                                                                                  |
| ----- | --------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| gt1   | NW drain system                               | active         |                                                                                                                                        |
| gt2   | Small mangrove island (W lagoon, upper)       | active         |                                                                                                                                        |
| gt3   | Elongated mangrove islet (W lagoon, lower)    | active         |                                                                                                                                        |
| gt4   | SE hammock island                             | active         |                                                                                                                                        |
| gt5   | Peninsula point (N tip of southern peninsula) | under_review   | Classification debatable; user research pending. Kept in file for traceability, excluded from scoring until resolved.                  |
| gt6   | E-shore trough                                | active         | Tier 3 trough; counts for matching.                                                                                                    |
| gt7   | Seagrass / sand / flat lobe (ambiguous)       | active         | `structure_type_options: ["seagrass_patch", "sand_lobe", "shallow_flat"]`. v3 is expected to surface this with `needs_deeper_zoom: true`; any of the three types is a hit. |
| gt8   | Central junction island                       | active         |                                                                                                                                        |

---

## Out-of-scope reminders

- **Phase D substructure work** (IDENTIFY / VALIDATE_CELLS / EXTRACT on z18 mosaics) is not in this card set. That path stays as-is.
- **Phase E validation framework** (pure findings, runs after B/C/D) — separate task set.
- **Phase G UI** (anchor-level accept/reject) — separate project.
- **F, H, I** — explicitly deferred post-v1 per `docs/PIPELINE_PHASES.md`.
