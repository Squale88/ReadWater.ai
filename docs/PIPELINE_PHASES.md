# ReadWater.ai Pipeline Phase Guide — v1

## Scope

v1 ships six phases:

- **Phase A** — Recursive Discovery
- **Phase B** — Context Assembly
- **Phase C** — Anchor Identification
- **Phase D** — Structure Decomposition
- **Phase E** — Phase-Level Validation
- **Phase G** — Human Correction (minimal: per-anchor accept/reject)

Deferred to later versions: **Phase F** (Editorial Audit), **Phase H** (Assessment / tagging), **Phase I** (Publish).

**v1 output:** human-approved structured knowledge — anchors with decomposed substructures, evidence, and provenance — written to disk. Downstream consumers arrive when F/H/I land.

---

## Core design principles

1. Separate coarse discovery from fine interpretation.
2. Separate structure identification from structure decomposition.
3. Treat context assembly as a first-class phase.
4. Preserve known facts rather than re-inferring them.
5. **Validation is pure: emits findings, never mutates.**
6. Human correction operates on structured objects, not text.
7. Provenance and traceability are required throughout.
8. v1 is forward-only: no automatic rerun loops.

---

## Cross-cutting rules

### Object state model

Every structured object (cell, context bundle, anchor, substructure) carries:

- `state` ∈ `draft | validated | approved | rejected`
- `phase_history` — ordered log of which phase produced or touched it, with timestamp
- `provenance` — source imagery refs, overlay refs, prompt id/version, provider config
- `findings` — unresolved validation findings attached to the object

State transitions:

- Phase A/B/C/D emit objects in `draft`.
- Phase E transitions `draft → validated`. Findings (if any) are attached but do not block the transition.
- Phase G transitions `validated → approved` or `validated → rejected`.

### Storage

Filesystem JSON under `data/areas/{area}/…`. No database in v1.

### Hash-addressable outputs

Each phase computes an input hash (bundle hash + prompt version + provider config + upstream object states) and stores it on the output. Caching is not implemented in v1, but the contract is hash-addressable so caching can be added later without reshaping interfaces.

### Known fact vs. inference

A **known fact** has a citable external source (NOAA ENC, FWC habitat layer, NAIP) or a prior human approval. A **model inference** is anything Claude derived from imagery, even at high confidence. Phase B tags every field as one or the other. Downstream phases must not conflate them.

### Silent reinterpretation rule

These fields cannot change after emission without a recorded event (human edit or explicit rerun):

- `anchor.id`, `anchor.type`, `anchor.center`
- `member.parent_anchor_id`, `subzone.parent_anchor_id`
- any `geometry.extent`

Later phases may add context or findings, but not rewrite these values.

### Feedback loops

v1 is forward-only. A rejected anchor drops from downstream consumption. Rejection does **not** trigger automatic rerun of C or D. If a human wants a corrected version of a rejected anchor, they re-run the relevant phase manually.

### Provenance requirements

Every structured object records: which phase emitted it, which imagery and overlays were used, which prompt id/version produced it, and (if touched by human) which user action changed it.

---

## Phase A — Recursive Discovery

**Purpose.** Narrow the starting region down to the zoom-16 cells worth analyzing further. This is a funnel, not a structure interpretation phase.

**What it does.**
- Recursively grid-scores cells from starting zoom (10 or 12) down to terminal zoom 16.
- Classifies each cell as fishable / not fishable with confidence and supporting notes.
- Recurses only into qualifying cells.
- **Threads lightweight parent context forward into child scoring** — preserves today's parent→child disambiguation (keeps A from promoting residential canals, retention ponds, isolated inland water, etc.).
- Preserves enough metadata for Phase B to rebuild a canonical context bundle.

**What it does not do.**
- No anchor identification.
- No substructure decomposition.
- No fishability scoring.
- No canonical context bundle — Phase B owns that.

**Inputs.** Source imagery for the current zoom, grid definitions, cell geometry, discovery prompt, exclusion rules.

**Outputs.** For each qualifying path to z16: cell id, zoom, parent lineage, bounds/center, fishable y/n + confidence, supporting notes, exclusion notes, references to all imagery used. Objects emitted in `draft`.

**Quality goals.** Fast, consistent, low variance, conservative about advancing weak cells, strong at excluding known false-positive types.

---

## Phase B — Context Assembly

**Purpose.** Build the canonical Z16 context bundle that Phases C and D consume. This is where imagery, metadata, ancestor context, overlays, and known facts are normalized into one coherent object.

**What it does.**
- Gathers local z16 imagery, parent and ancestor imagery, positional-context imagery.
- Ingests Phase A's threaded context and discovery notes.
- Ingests and aligns overlays: NOAA ENC channels, FWC oyster/seagrass, NAIP 4-band, water/land masks.
- Tags each field as **known fact** or **inference**.
- Summarizes what the context implies without overstating certainty.
- Preserves raw references and provenance for audit and human review.

**What it does not do.**
- No anchor identification.
- No fishability interpretation.
- Does not treat overlays as unquestionable truth — overlays are evidence with their own confidence and limitations.

**Inputs.** Qualified z16 cells from Phase A, their parent/ancestor imagery refs, overlay sources, discovery notes, any known exclusions or uncertainty flags.

**Outputs.** A `Z16ContextBundle` per qualified cell containing:

- cell identity and geometry
- local/ancestor/positional image refs
- discovery notes
- overlay summaries + raw refs
- normalized evidence objects, each tagged `known_fact` or `inference`
- uncertainty flags
- full provenance

**Quality goals.** Complete, explicit about what's known vs. inferred, honest about uncertainty, traceable.

---

## Phase C — Anchor Identification

**Purpose.** Identify the dominant organizing structures (anchors) in the z16 cell, and plan the z18 capture for decomposition.

**What it does.**
- Consumes the `Z16ContextBundle`.
- Identifies major anchors (creek mouths, feeder drains, trough edges, shoreline points, island tips, oyster spines, channel-flat confluences, major shoreline bends, structure-linked depressions).
- For each anchor: type, label, rough center, rough extent, confidence, why it matters, supporting evidence, relationship to overlays, priority rank.
- **Emits a seed z18 fetch plan per anchor.** Phase D is allowed to expand the plan within a bounded tile budget if the anchor runs off-frame. Phase C does not pre-over-fetch defensively.

**What it does not do.**
- No full substructure decomposition.
- No fishability scoring.
- No final geometry — C's extents are seeds, not commitments.

**Inputs.** `Z16ContextBundle`.

**Outputs.** A small, high-quality set of `AnchorStructure` objects, each with a seed `Z18FetchPlan`. Emitted in `draft`.

**Quality goals.** Small, high-signal anchor list. Identify organizers, not every possible feature.

---

## Phase D — Structure Decomposition

**Purpose.** Use z18 imagery to resolve each anchor into fishable substructures and local complexes.

**What it does.**
- For each anchor, executes the seed z18 fetch plan from C.
- **Allowed to expand the fetch plan within a bounded tile budget** (the current `RESOLVE_CONTINUATION` behavior) when the anchor clearly runs off-frame. Expansions are logged in `phase_history`.
- Assembles z18 mosaics where needed.
- Identifies substructures (feeder cuts, trough fingers, ambush corners, oyster nodes, seams, depressions, pinch points, edge transitions, current breaks, flat-to-channel transitions).
- Defines influence zones around anchors.
- Ties every substructure to a parent anchor — no orphans.

**What it does not do.**
- Does not rewrite anchor identity or type. If D concludes C was wrong, it raises a finding (surfaced in G); it does not silently mutate the anchor.
- No fishability scoring.
- No tag generation — deferred to Phase H.

**Inputs.** Validated anchor objects + seed `Z18FetchPlan`, relevant carry-forward context from the `Z16ContextBundle`.

**Outputs, per anchor.** `LocalComplex`, `InfluenceZone`, `FishableSubzone`, refined evidence, decomposition notes, geometry, confidence, provenance. Emitted in `draft`.

**Quality goals.** Detailed but disciplined. Every substructure grounded in visible or overlay-supported evidence and tied to an anchor.

---

## Phase E — Phase-Level Validation

**Purpose.** Strict schema and consistency checks after B, C, and D. **Pure: emits findings, never mutates.** No auto-repair in v1.

**Where it runs.** After Phase B, after Phase C, after Phase D. Each run validates only what that phase produced.

**What it checks.**
- Schema and required fields present.
- Parent-child relationships resolve.
- Geometry sanity: extent inside declared bounds; no zero-area or self-intersecting polygons.
- Required evidence present on high-confidence claims.
- Provenance complete.
- No orphan subzones or influence zones.
- Anchor centers inside the z16 cell.
- Subzone extents inside parent anchor extent (or carry an explicit note).

**What it does not do.**
- Does not edit, normalize, merge, or auto-correct anything.
- Does not block phase transitions. Findings are attached to objects; the object moves from `draft → validated` regardless of finding severity.
- Does not re-run interpretive model calls.

**Findings format.** `{issue_code, severity, object_id, field, message, recommended_action}`. Recommended actions are advisory only in v1.

**Outputs.** Validation report per phase run; findings attached to each object.

**Quality goals.** Fast, deterministic, rule-based. If you can't write it as a rule, it's not Phase E's job.

---

## Phase G — Human Correction (minimal)

**Purpose.** Give a human an efficient way to approve or reject anchors before content is consumed downstream.

**Granularity (v1).** **Per-anchor accept/reject only.** When an anchor is rejected, all its substructures (members, subzones, influence zones) are dropped with it. Substructure-level editing is deferred.

**What the UI shows per anchor.**
- Anchor label, type, center, extent on imagery.
- All substructures tied to the anchor.
- Z16 and z18 imagery used.
- Overlay evidence.
- Validation findings from Phase E (non-blocking, displayed for context).
- Provenance summary.

**What the human can do (v1).**
- Accept anchor → state becomes `approved`; substructures inherit.
- Reject anchor → state becomes `rejected`; substructures inherit.
- Add a free-text note.

**What the human cannot do in v1.**
- Edit geometry.
- Edit type or label.
- Merge or split anchors.
- Act on individual substructures independent of the parent anchor.

All of those are future work. If a human wants a corrected version of a rejected anchor, they re-run C/D manually.

**Outputs.** Per anchor: final `state`, human note, decision timestamp, reviewer id.

---

## Operational flow (v1)

1. Run Phase A (recursive discovery) → qualified z16 cells.
2. For each qualified z16 cell, run Phase B (context assembly) → `Z16ContextBundle`.
3. Run Phase E on the bundle.
4. Run Phase C (anchor identification) against the bundle → anchors + seed z18 fetch plans.
5. Run Phase E on anchors.
6. For each anchor, run Phase D (decomposition) → substructures.
7. Run Phase E on decomposition outputs.
8. Surface anchors + substructures + validation findings in the Phase G UI.
9. Human accepts or rejects each anchor.
10. Approved anchors + their substructures become v1's final output: human-approved structured knowledge on disk.

---

## Explicitly deferred (post-v1)

- **Phase F — Editorial Audit.** Cross-object reconciliation, duplication detection, normalized labeling, plausibility review.
- **Phase H — Assessment (tagging).** Condition-independent tags (tide stage, wind octant, season, etc.) plus confidence, for the planning engine to query. Requires a controlled tag vocabulary designed from the planning-engine query backwards.
- **Phase I — Publish.** State promotion and versioned payloads for downstream consumers.
- Substructure-level human review, geometry editing, merge/split operations.
- Validator auto-repair.
- Automatic rerun / invalidation on rejection.
- Database-backed storage and version history.
- Output caching keyed by input hash.

---

## What Claude should use this guide for

- Treat each phase as a distinct contract with explicit inputs and outputs.
- Preserve phase separation — especially C vs. D, and B vs. everything.
- Keep Phase E pure — no mutation, no auto-repair.
- Preserve provenance and known-fact/inference distinction at every step.
- Anchor-level human approval in G is the gate that produces v1's final output; nothing downstream of G exists in v1.
