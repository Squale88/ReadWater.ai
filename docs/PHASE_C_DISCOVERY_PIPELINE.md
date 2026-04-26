# Phase C — Unified Anchor Discovery Pipeline

Plan for the production-side wire-up of v3 + coord-gen + habitat evidence
into a single config-driven discovery pipeline that replaces the legacy
z16 grid-cell `DISCOVER` path in `src/readwater/pipeline/structure/agent.py`.

This supersedes the original TASK-6 spec in `docs/PHASE_C_TASKS.md` because
the eval work (sweeps + tier-weighted scoring) revealed two things that
weren't anticipated:

1. The model handles the 8×8 grid inconsistently — sometimes correctly,
   sometimes by hallucinating extended labels. Forcing one mode in
   production foregoes useful information either way.
2. Coord-gen's pixel placement quality is the system's weak link, not v3
   identification. Both stages need their own grid/nogrid switch so we can
   tune them independently.

## Goal

A single `run_anchor_discovery(bundle, config) -> StructurePhaseResult`
function that, for one z16 cell:

1. **DISCOVER** — runs v3 in the configured mode(s)
2. **COORDS** — runs coord-gen in the configured mode(s) over v3's anchors
3. **PLAN_CAPTURE** — derives a `Z18FetchPlan` for each placed anchor
4. **ASSEMBLE** — emits `AnchorStructure[]` in `state="draft"` with full
   `provenance`, `phase_history`, `findings`, and seed fetch plans

Habitat evidence (NOAA channels, FWC oyster/seagrass, NAIP water/land) is
optionally injected into the v3 prompts as supporting context.

## Configuration

```python
@dataclass(frozen=True)
class AnchorDiscoveryConfig:
    # v3 prompt variant.
    #   "nogrid"     - clean z16 image; v3 has no grid_cells field
    #   "grid"       - 8x8 A1-H8 overlay; v3 emits grid_cells
    #   "comparison" - run both, save both, pick winner per below
    v3_mode: Literal["nogrid", "grid", "comparison"] = "nogrid"

    # coord-gen prompt variant.
    #   "nogrid"     - clean z16 image; pure compass+visual reasoning
    #   "grid"       - 8x8 overlay; coord-gen can reference cells
    #   "comparison" - run both, save both, pick winner per below
    coords_mode: Literal["nogrid", "grid", "comparison"] = "grid"

    # When *_mode is "comparison", which output is canonical for downstream
    # consumers (PLAN_CAPTURE, AnchorStructure assembly)?
    v3_comparison_winner: Literal["nogrid", "grid"] = "nogrid"
    coords_comparison_winner: Literal["nogrid", "grid"] = "grid"

    # Habitat evidence (TASK-3) — injected into the v3 user prompt as a
    # supporting evidence table. Off by default during initial rollout.
    inject_evidence: bool = False

    # Z18 tile plan parameters — passed to mosaic.z18_tile_plan_from_latlon
    tile_budget_z18: int = 25

    # Coord-gen failure policy (per addendum in PHASE_C_TASKS.md):
    #   - per-anchor failures (out-of-bounds, low confidence) → keep with Finding
    #   - whole-batch malformed JSON → fail the COORDS stage
    # No knobs here, just documented.
```

The default config above (`v3_mode=nogrid`, `coords_mode=grid`,
`inject_evidence=False`) reflects what the sweep data showed worked best:
v3 in nogrid avoids the grid-hallucination tax, coord-gen with grid uses
the overlay as a useful spatial reference. Evidence injection stays off
until TASK-3 ships and is A/B'd.

`comparison` mode is opt-in and doubles API cost; designed for production
sampling and offline comparison, not the default path.

## Pipeline flow

```
                                ┌──────────────────────────┐
                                │ Z16ContextBundle         │
                                │ (z16 + ancestors + bundle│
                                │  json + optional evidence)│
                                └──────────┬───────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────┐
                       │ DISCOVER (anchor_discovery)        │
                       │   - load v3_<mode>_{system,user}   │
                       │   - send 4-image handoff + bundle  │
                       │     (+ evidence_table if enabled)  │
                       │   - if mode=comparison: 2 calls    │
                       │   - parse anchors[]                 │
                       │   - emit DISCOVER PhaseEvent        │
                       └──────────┬─────────────────────────┘
                                  │ anchors[] (no pixel data)
                                  ▼
                       ┌────────────────────────────────────┐
                       │ COORDS (anchor_discovery)          │
                       │   - load coords_<mode>_{sys,user}  │
                       │   - send image (clean or grid) +   │
                       │     anchor list                    │
                       │   - if mode=comparison: 2 calls    │
                       │   - parse pixel placements         │
                       │   - per-anchor validation:         │
                       │     * out-of-bounds → Finding(warn)│
                       │     * low confidence → Finding(info)│
                       │   - whole batch malformed → fail   │
                       │   - emit COORDS PhaseEvent          │
                       └──────────┬─────────────────────────┘
                                  │ pixel_center, pixel_bbox per anchor
                                  ▼
                       ┌────────────────────────────────────┐
                       │ PLAN_CAPTURE (anchor_discovery)     │
                       │   - pixel_center → anchor_center_   │
                       │     latlon via geo.pixel_to_latlon │
                       │   - pixel_bbox → rough_extent_m     │
                       │   - mosaic.z18_tile_plan_from_      │
                       │     latlon(...) → Z18FetchPlan     │
                       │   - skip anchors with no coords     │
                       │   - emit PLAN_CAPTURE PhaseEvent    │
                       └──────────┬─────────────────────────┘
                                  │
                                  ▼
                       ┌────────────────────────────────────┐
                       │ ASSEMBLE                           │
                       │   - build AnchorStructure[] with:  │
                       │     state="draft"                  │
                       │     provenance (real)              │
                       │     phase_history (3 events)       │
                       │     findings (forwarded)           │
                       │     seed_z18_fetch_plan            │
                       │     priority_rank (from v3 order)  │
                       │     zone_id (from v3 zone_id)      │
                       │   - return StructurePhaseResult    │
                       └────────────────────────────────────┘
```

## Files

### New

- **`src/readwater/pipeline/structure/anchor_discovery.py`** — top-level
  module. Holds `AnchorDiscoveryConfig`, `discover_anchors_v3`,
  `locate_anchor_coords`, `plan_capture_for`, `run_anchor_discovery`.
  Pure orchestration; LLM and geo logic live in the modules they already do.

- **`prompts/anchor_coords_grid_system.txt`** — rename of current
  `anchor_coords_system.txt`.
- **`prompts/anchor_coords_grid_user.txt`** — rename of current
  `anchor_coords_user.txt`.
- **`prompts/anchor_coords_nogrid_system.txt`** — new. Clean-image variant.
  No "8×8 overlay" references; instructs the model to reason from compass
  position + visible landmarks only. Same JSON output schema.
- **`prompts/anchor_coords_nogrid_user.txt`** — new. Same input shape as
  the grid user prompt minus the grid mention.

- **`tests/test_anchor_discovery.py`** — unit tests for config dispatch,
  failure-policy handling, AnchorStructure assembly. Integration test
  using cached LLM outputs (no API calls), seeded from one of the existing
  per-cell `review/` artifact sets.

### Modified

- **`prompts/anchor_identification_v3_nogrid_system.txt`** — TASK-3 evidence
  block added.
- **`prompts/anchor_identification_v3_grid_system.txt`** — same.
- **`prompts/anchor_identification_v3_nogrid_user.txt`** — `{evidence_table}`
  placeholder.
- **`prompts/anchor_identification_v3_grid_user.txt`** — same.
- **`src/readwater/pipeline/structure/prompts.py`** — wrapper functions for
  the four prompt pairs (v3 nogrid/grid, coords nogrid/grid). Unifies how
  the agent calls into prompts.
- **`src/readwater/pipeline/structure/agent.py`** — replace the legacy
  z16 grid-cell `DISCOVER` block (lines ~103–193) with a call to
  `anchor_discovery.run_anchor_discovery(...)`. The downstream IDENTIFY /
  EXTRACT path for substructures (Phase D) stays unchanged.
- **`src/readwater/pipeline/structure/__init__.py`** — export
  `AnchorDiscoveryConfig`, `run_anchor_discovery`.

### Deleted (TASK-7 cleanup, after TASK-6 is green)

- `prompts/anchor_identification_system_v2.txt`,
  `anchor_identification_user_v2.txt` — superseded by v3 nogrid/grid pairs.
- `prompts/discover_anchors_{system,user}.txt` — legacy discovery prompts.
- Legacy DISCOVER helpers in `agent.py` (whatever isn't called after the
  rewrite).
- `prompts/grid_scoring_user*.txt` consolidation per original TASK-7.

## Comparison mode handling

When `v3_mode="comparison"`:

1. Run v3 nogrid → save `anchors_v3_nogrid.json` to the run dir
2. Run v3 grid → save `anchors_v3_grid.json` to the run dir
3. Save a `discover_comparison.json` listing anchors-only-in-A,
   anchors-only-in-B, agreed (matched by structure_type + position
   keyword overlap)
4. Pick `v3_comparison_winner` and feed those anchors to COORDS

When `coords_mode="comparison"`:

1. Run coord-gen nogrid on the chosen v3 anchors → save
2. Run coord-gen grid on the same v3 anchors → save
3. Save a `coords_comparison.json` with per-anchor pixel deltas between
   the two runs (sanity check on consistency)
4. Pick `coords_comparison_winner` and feed those placements to PLAN_CAPTURE

The downstream `AnchorStructure[]` always reflects the winners. Both raw
outputs persist in the run dir for offline review using the same
`<cell>_v3_raw_response.md` / `<cell>_coordgen_raw_response.md` naming
the sweep harness already uses.

## Provenance population

Each emitted `AnchorStructure.provenance` carries:

```python
Provenance(
    source_images=[z16_image_path, z15_path, z14_path, z12_path,
                   coords_image_path],   # records which image coord-gen used
    overlay_refs=[grid_overlay_path] if any_grid_used else [],
    prompt_id="anchor_identification_v3",
    prompt_version=f"v3_{v3_mode}+coords_{coords_mode}",
    provider_config={"model": MODEL, "max_tokens": MAX_TOKENS, ...},
    input_hash=hash(image_bytes + prompt_text + bundle_json),
)
```

When `comparison` mode picks a winner, `prompt_version` records the winner
(e.g. `"v3_nogrid+coords_grid"`). The losing run's bytes are still on disk
under the run dir for replay/audit.

## Phase history

Each emitted `AnchorStructure.phase_history` contains, in order:

```python
[
  PhaseEvent(phase="C.DISCOVER", action="emit", actor="anchor_discovery.v3_<mode>",
             timestamp=..., note="anchor #N of M from v3"),
  PhaseEvent(phase="C.COORDS", action="locate", actor="anchor_discovery.coords_<mode>",
             timestamp=..., note="pixel_center=(x,y) conf=0.85"),
  PhaseEvent(phase="C.PLAN_CAPTURE", action="plan", actor="mosaic.z18_tile_plan",
             timestamp=..., note="9 tiles, 1029m extent"),
]
```

## Failure handling (locked policies)

From `docs/PHASE_C_TASKS.md` addendum:

- **Per-anchor coord-gen failure** (out-of-bounds, very low
  `placement_confidence`): keep the anchor, set `seed_z18_fetch_plan=None`,
  attach a `Finding(severity="warn")`. Phase E surfaces. No raise.
- **Whole-batch coord-gen JSON malformed**: raise from COORDS stage. The
  cell's `StructurePhaseResult.findings` records the failure; no anchors
  for this cell get fetch plans.
- **v3 returns zero anchors**: not an error — emit empty
  `StructurePhaseResult.anchors` with a `Finding(severity="info")`.
- **Strict matching of coord-gen output to v3 input** by `anchor_id`. No
  fuzzy fallback. Unmatched coord-gen entries log + drop.

## Migration

The existing `provenance` requirement on `AnchorStructure` is hard. Per
the addendum, legacy cached anchor JSON gets backfilled by a one-shot
script (TASK-4 spec). When the new `run_anchor_discovery` runs in
production, **all new anchors carry real provenance**; nothing legacy
should be passing through this path.

## Tests

- `test_anchor_discovery_config_defaults` — config defaults match doc
- `test_anchor_discovery_v3_dispatch` — config selects the right prompt pair
- `test_anchor_discovery_coords_dispatch` — same for coord-gen
- `test_anchor_discovery_keep_and_flag_on_oob` — out-of-bounds pixel produces
  Finding + `seed_z18_fetch_plan=None`
- `test_anchor_discovery_fail_stage_on_bad_batch` — malformed coord-gen
  JSON raises from COORDS
- `test_anchor_discovery_strict_anchor_id_matching` — unmatched coord-gen
  entries don't poison the placement output
- `test_anchor_discovery_assembles_phase_history` — emitted anchors carry
  3 events in order
- `test_anchor_discovery_comparison_mode` — both variants run, winner
  selected, both raw responses persisted
- `test_anchor_discovery_integration_root_10_8` — full flow on a cached
  fixture (no API), asserts ≥3 anchors with non-null lat/lon centers and
  populated `seed_z18_fetch_plan.tile_centers`

## Build order (dependencies)

```
1. coord-gen nogrid prompt pair  (S — copy + edit existing grid prompts)
            │
2. TASK-3 evidence injection      (S–M — prompt edits + harness wiring)
            │
3. anchor_discovery.py module      (M — orchestration + assembly)
            │
4. TASK-6 wire into agent.py       (M — replace legacy DISCOVER block)
            │
5. tests + integration             (M — unit + cached-fixture integration)
            │
6. TASK-7 cleanup                  (M — delete legacy paths)
```

Steps 1–3 can run in parallel-ish; 4 needs all three. 5 gates 6.

## Out of scope (explicit)

- **Phase D substructure work** (z18 IDENTIFY/EXTRACT). Stays as-is.
- **Phase E validation framework** — separate, runs after this.
- **Eval harness** (sweep_anchor_pipeline.py + per-cell review/) — keep
  for tuning but not part of the production path.
- **Comparison-mode merge logic** — only "pick a winner" is in scope. A
  future task could add real merge (anchors agreed by both, etc.).
