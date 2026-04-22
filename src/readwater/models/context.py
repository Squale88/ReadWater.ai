"""Retained-cell context and z16 handoff bundle models.

Phase 1 of the retained-cell-context work. These models store typed
observations, inferences, feature hypotheses, and open questions produced
during recursive analysis, plus the visual/context package compiled at the
z16 handoff to the structure phase.

Design notes:
- Identity is by deterministic IDs: observations/morphology/threads/questions
  all use f"{cell_id}:<kind>:<n>" patterns so cross-references across cells
  and across zoom levels are stable.
- The Z16ContextBundle uses three decoupled keyed surfaces:
    visuals   keyed by VisualRole
    contexts  keyed by cell_id
    evidence  keyed by cell_id
  `lineage` is an ordered list for traversal only; it is never used for
  positional alignment with visuals/contexts/evidence.
- z15_same_center is VISUAL-ONLY and deliberately has no CellContext entry.
- This phase does not touch structure-phase code. The bundle is produced
  and persisted to disk for later consumption.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from readwater.models.cell import BoundingBox


# --- Role vocabulary ---


class VisualRole(str, Enum):
    """Role of an image in the z16 handoff bundle.

    Role names bind both relative lineage position and zoom so consumers
    never have to infer which image is which. The closed set covers the
    two supported start zooms (z10 and z12).
    """

    Z16_LOCAL = "z16_local"                          # exact tile being assessed
    Z15_SAME_CENTER = "z15_same_center"              # auxiliary local continuity; NOT a lineage member
    Z14_PARENT = "z14_parent"                        # recursive parent
    Z12_GRANDPARENT = "z12_grandparent"              # recursive grandparent (root for z12-start)
    Z10_GREAT_GRANDPARENT = "z10_great_grandparent"  # recursive great-grandparent (root for z10-start)


# --- Lineage reference ---


class LineageRef(BaseModel):
    """A compact reference to one cell in the recursive lineage chain.

    Used both as in-memory state carried through recursion and as the
    serialized lineage entries in Z16ContextBundle.
    """

    cell_id: str = Field(description="Tree-path cell ID, e.g. 'root', 'root-6', 'root-6-11'")
    zoom: int = Field(ge=0, description="Google Maps zoom level")
    depth: int = Field(ge=0, description="Recursion depth (0 = root)")
    center: tuple[float, float] = Field(description="(lat, lon) center of the cell")
    bbox: BoundingBox = Field(description="Geographic bounding box of this cell")
    image_path: str | None = Field(
        default=None,
        description="Path to this cell's existing base image (unannotated). None if not yet fetched",
    )
    position_in_parent: tuple[int, int] | None = Field(
        default=None,
        description="(row, col) within the parent's 4x4 grid; None for the root cell",
    )


# --- Context atoms ---


class DirectObservation(BaseModel):
    """A literal visible feature at this zoom — 'what I see, without inference'."""

    observation_id: str = Field(
        description="Deterministic ID, e.g. f'{cell_id}:obs:{n}'",
    )
    label: str = Field(
        description="Short label for the feature (e.g. 'mangrove_shoreline', 'tidal_inlet')",
    )
    location_hint: str = Field(
        default="",
        description="Short positional phrase: 'S edge', 'cells C3-D4', 'NE quadrant'",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class MorphologyInference(BaseModel):
    """A morphology/connectivity inference drawn from the current image plus
    the ancestor chain. Guidance, not truth — downstream should verify."""

    inference_id: str = Field(
        description="Deterministic ID, e.g. f'{cell_id}:morph:{n}'",
    )
    kind: str = Field(
        description="Relation kind: 'enclosed_by' | 'drains_to' | 'tidal_exchange_via' | "
        "'barrier_between' | 'continuous_with' | other",
    )
    statement: str = Field(description="One-sentence claim")
    references: list[str] = Field(
        default_factory=list,
        description="Other observation_ids or ancestor cell_ids this claim depends on",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CandidateFeatureThread(BaseModel):
    """A fishing-relevant hypothesis that may be worth resolving at deeper zoom.

    Threads can chain across zoom levels via parent_thread_id; that is how a
    z12 hypothesis becomes supported/contradicted/resolved at z14 or z16.
    """

    thread_id: str = Field(
        description="Deterministic ID, e.g. f'{cell_id}:th:{n}'",
    )
    feature_type: str = Field(
        description="Candidate feature type, aligned with AnchorStructure.structure_type vocabulary",
    )
    status: str = Field(
        description="'hypothesized' | 'supported' | 'contradicted' | 'resolved'",
    )
    summary: str = Field(default="")
    supporting_observation_ids: list[str] = Field(default_factory=list)
    parent_thread_id: str | None = Field(
        default=None,
        description="Thread ID from an ancestor's context that this thread derives from",
    )
    needs_zoom: int | None = Field(
        default=None,
        description="Zoom level expected to resolve this thread",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.3)


class UnresolvedQuestion(BaseModel):
    """A question carried forward so a deeper-zoom pass can attempt resolution."""

    question_id: str = Field(
        description="Deterministic ID, e.g. f'{cell_id}:q:{n}'",
    )
    question: str
    target_zoom: int | None = Field(
        default=None,
        description="Zoom level expected to be able to resolve the question",
    )


class EvidenceSummary(BaseModel):
    """Per-layer summary of surveyed ground truth at a cell.

    Per-grid-cell detail stays in pipeline.evidence for direct structure-
    phase consumption. This is a whole-cell digest suitable for the bundle.
    """

    layer: str = Field(description="'water' | 'channel' | 'oyster' | 'seagrass'")
    mask_path: str | None = Field(default=None)
    coverage_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Whole-cell coverage fraction in [0, 1]",
    )
    notes: str = Field(default="")


class CellContext(BaseModel):
    """Typed context produced for any retained cell after grid scoring.

    Separates direct observations, morphology inferences, candidate feature
    threads, open questions, and evidence. Consumers should never collapse
    these into a single narrative blob.
    """

    cell_id: str
    zoom: int = Field(ge=0)
    observations: list[DirectObservation] = Field(default_factory=list)
    morphology: list[MorphologyInference] = Field(default_factory=list)
    feature_threads: list[CandidateFeatureThread] = Field(default_factory=list)
    open_questions: list[UnresolvedQuestion] = Field(default_factory=list)
    evidence: list[EvidenceSummary] = Field(default_factory=list)
    model_used: str = Field(default="")
    source_images: list[str] = Field(
        default_factory=list,
        description="Image paths fed to the generator (for audit)",
    )
    raw_response_path: str | None = Field(
        default=None,
        description="Path to the raw LLM markdown response saved alongside",
    )


# --- Visual references ---


class VisualContextRef(BaseModel):
    """One image in the z16 handoff bundle, with explicit role and contents.

    Two distinct bboxes are modeled separately on purpose:
      depicts_bbox           = the geographic coverage this image actually shows
      overlay_footprint_bbox = the inner rectangle drawn on the overlay

    A consumer must never conflate these. `depicts_bbox` answers "what am I
    looking at?"; `overlay_footprint_bbox` answers "what rectangle is drawn
    on me?".
    """

    role: VisualRole
    zoom: int = Field(ge=0)
    center: tuple[float, float]

    depicts_bbox: BoundingBox = Field(
        description="Geographic coverage this image actually shows",
    )
    overlay_footprint_bbox: BoundingBox | None = Field(
        default=None,
        description="Inner rectangle drawn on the overlay image. "
        "None when no overlay is generated (e.g. z16_local).",
    )

    base_image_path: str = Field(
        description="Existing on-disk image; never mutated by this phase",
    )
    overlay_image_path: str | None = Field(
        default=None,
        description="Overlay file written at bundle-assembly time. "
        "None when no overlay applies to this role.",
    )
    overlay_draws: list[str] = Field(
        default_factory=list,
        description="Labels of footprints drawn on the overlay (e.g. ['z16_footprint'])",
    )
    lineage_cell_id: str | None = Field(
        default=None,
        description="cell_id of the lineage member this image belongs to. "
        "None for z15_same_center since it is not a recursive lineage member.",
    )


# --- The bundle ---


class Z16ContextBundle(BaseModel):
    """The compiled context package for a z16 cell, written to disk at handoff.

    Three keyed surfaces (visuals, contexts, evidence) decouple the bundle
    from positional list alignment. `lineage` is ordered for traversal only.

    Layer counts for the two supported start zooms:
      start_zoom=12 -> lineage has 3 entries (z12 root, z14 parent, z16 self)
                       visuals has 4 entries (z16_local, z15_same_center,
                                              z14_parent, z12_grandparent)
                       contexts has 3 entries (recursive lineage members)
      start_zoom=10 -> lineage has 4 entries (z10 root, z12, z14, z16 self)
                       visuals has 5 entries (+ z10_great_grandparent)
                       contexts has 4 entries
    """

    cell_id: str = Field(description="The z16 cell being handed off")
    compiled_at: str = Field(description="ISO timestamp")
    schema_version: str = Field(default="1.0")

    lineage: list[LineageRef] = Field(
        description="Ordered root-most ancestor -> z16 self, inclusive",
    )
    visuals: dict[VisualRole, VisualContextRef] = Field(
        description="Image set keyed by role",
    )
    contexts: dict[str, CellContext] = Field(
        description="cell_id -> typed context; only recursive lineage members (no z15)",
    )
    evidence: dict[str, list[EvidenceSummary]] = Field(
        default_factory=dict,
        description="cell_id -> per-layer evidence; keyed like contexts. "
        "Mostly empty in Phase 1 at non-z16 levels.",
    )
