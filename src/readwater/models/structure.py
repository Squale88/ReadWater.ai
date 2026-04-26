"""Structure-phase zone models.

Two geometry classes distinguish the provenance of a shape:

- ObservedGeometry: the boundary was derived from a geometry extractor
  operating on the image (seed points + extractor algorithm). This is the
  output of the EXTRACT state.
- InterpretedGeometry: the boundary was drawn or derived from LLM reasoning
  (an influence zone polygon, or an envelope computed from member features).
  This is the output of the INTERPRET state.

Zone objects reference the kind of geometry appropriate to them:
  AnchorStructure, FishableSubzone, MemberFeature  → ObservedGeometry
  InfluenceZone, LocalComplex.envelope             → InterpretedGeometry

LocalComplex is NOT a single union polygon. It is a collection of member
features, each with its own ObservedGeometry, plus an optional interpreted
envelope.

Phase C v1 additions (PhaseEvent, Provenance, Finding, Z18FetchPlan, plus the
state/phase_history/provenance/findings/seed_z18_fetch_plan/priority_rank/
zone_id fields on AnchorStructure) bring AnchorStructure up to the contract
in docs/PIPELINE_PHASES.md. See docs/PHASE_C_TASKS.md TASK-4 for the spec
and the addendum for the "backfill marker, not regenerate" migration policy
on legacy cached anchor JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Literal aliases for state machines ---

StructureState = Literal["draft", "validated", "approved", "rejected"]
FindingSeverity = Literal["info", "warn", "error"]


# --- Geometry classes ---


class ObservedGeometry(BaseModel):
    """A polygon extracted from image observation via grid cells or seed points."""

    pixel_polygon: list[tuple[int, int]] = Field(
        description="Polygon vertices in pixel space of the source image",
    )
    latlon_polygon: list[tuple[float, float]] = Field(
        description="Polygon vertices in (lat, lon), same order as pixel_polygon",
    )
    image_ref: str = Field(description="Name of the source image (e.g. 'mosaic')")
    extractor: str = Field(
        description="Name of the extractor that produced this polygon "
        "(e.g. 'gridcell', 'clickbox', 'sam2_region', 'fallback')",
    )
    extraction_mode: str = Field(
        description="Mode used: 'region' | 'corridor' | 'point_feature' | 'edge_band'",
    )
    seed_cells: list[str] = Field(
        default_factory=list,
        description="Grid-cell labels Claude selected (e.g. ['C3', 'C4', 'D3']). "
        "Empty when extraction was non-grid-based.",
    )
    grid_rows: int | None = Field(
        default=None,
        description="Rows of the grid Claude saw, when extractor='gridcell'",
    )
    grid_cols: int | None = Field(
        default=None,
        description="Columns of the grid Claude saw, when extractor='gridcell'",
    )
    seed_positive_points: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Positive seed points (pixel coords). For gridcell extraction, "
        "these are the centroids of selected cells.",
    )
    seed_negative_points: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Negative seed points (pixel coords). For gridcell extraction, "
        "centroids of cells explicitly flagged as outside the feature.",
    )
    confidence: float | None = Field(
        default=None,
        description="Extractor-reported quality, if supported",
    )


class InterpretedGeometry(BaseModel):
    """A polygon derived from LLM reasoning, not image segmentation."""

    pixel_polygon: list[tuple[int, int]] = Field(
        description="Polygon vertices in pixel space of the source image",
    )
    latlon_polygon: list[tuple[float, float]] = Field(
        description="Polygon vertices in (lat, lon), same order as pixel_polygon",
    )
    image_ref: str = Field(description="Name of the source image (e.g. 'mosaic')")
    source: str = Field(
        description="Where the polygon came from: 'llm_polygon' | "
        "'convex_hull_of_anchor' | 'members_envelope'",
    )
    rationale: str = Field(default="", description="Optional LLM rationale for the shape")


# --- Phase C v1 envelope objects ---


class PhaseEvent(BaseModel):
    """One transition in an object's phase history.

    Append-only audit log. Every state change in the structure-phase pipeline
    (DISCOVER -> COORDS -> PLAN_CAPTURE -> ... ) records one PhaseEvent.
    """

    phase: str = Field(description="e.g. 'C.DISCOVER', 'C.COORDS', 'C.PLAN_CAPTURE'")
    action: str = Field(description="e.g. 'emit', 'update', 'reject', 'approve'")
    actor: str = Field(description="Subsystem or human that triggered the event")
    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    note: str | None = Field(default=None, description="Optional human-readable detail")


class Provenance(BaseModel):
    """Where this object came from. Required on every Phase-C output object.

    Migration: legacy cached anchor JSON predates this field. The TASK-4
    backfill script writes Provenance entries marked
    `prompt_version="legacy_pre_v1"` and `prompt_id="unknown"` so legacy
    anchors load cleanly without lying about their origin.
    """

    source_images: list[str] = Field(
        default_factory=list,
        description="Paths/refs of images the LLM call saw",
    )
    overlay_refs: list[str] = Field(
        default_factory=list,
        description="Paths/refs of any overlays drawn on those images (grid, ROI, etc.)",
    )
    prompt_id: str = Field(description="Stable identifier of the prompt pair used")
    prompt_version: str = Field(
        description="Version tag of the prompt pair (e.g. 'v3_nogrid', 'v3_grid', "
        "'legacy_pre_v1' for backfilled migrations)",
    )
    provider_config: dict = Field(
        default_factory=dict,
        description="Model/provider settings (model name, temperature, max_tokens, ...)",
    )
    input_hash: str = Field(
        default="",
        description="Hash of the inputs (image bytes + prompt text + bundle JSON), "
        "used for caching and replay matching",
    )


class Finding(BaseModel):
    """A non-fatal observation produced by validation, classification, or
    geometry steps. Findings are how Phase-C objects communicate problems
    without raising — Phase E surfaces them to the user."""

    issue_code: str = Field(description="Stable, machine-readable code, e.g. 'COORDS_OUT_OF_BOUNDS'")
    severity: FindingSeverity = Field(description="'info' | 'warn' | 'error'")
    object_id: str = Field(description="ID of the object the finding refers to")
    field: str | None = Field(default=None, description="Specific field on the object, when relevant")
    message: str = Field(description="Human-readable description")
    recommended_action: str | None = Field(
        default=None,
        description="Optional next step (e.g. 'rerun coord-gen', 'human review')",
    )


class Z18FetchPlan(BaseModel):
    """Tile-fetch plan for the Phase D z18 mosaic build.

    Produced by `mosaic.z18_tile_plan_from_latlon()` (TASK-5) at PLAN_CAPTURE
    time, then consumed by Phase D when it actually fetches the tiles.
    """

    tile_centers: list[tuple[float, float]] = Field(
        default_factory=list,
        description="(lat, lon) centers in row-major order",
    )
    tile_budget: int = Field(
        default=25,
        description="Hard cap on tile count for this anchor",
    )
    extent_meters: float = Field(
        default=0.0,
        description="Effective ground coverage of the planned mosaic, in meters per side",
    )


# --- Zone objects ---


class AnchorStructure(BaseModel):
    """The main visible organizing feature of a fishing picture. Observed.

    Phase C v1 fields (state, phase_history, provenance, findings,
    seed_z18_fetch_plan, priority_rank, zone_id) bring AnchorStructure up to
    the v1 Phase C contract in docs/PIPELINE_PHASES.md. Per TASK-4 + addendum,
    `provenance` is required on new instances. Legacy cached anchors get a
    backfilled `prompt_version="legacy_pre_v1"` Provenance via a one-shot
    migration script rather than silently defaulting to empty.
    """

    # Allow Pydantic to accept extra fields when loading older JSON snapshots
    # so we don't trip on small additions during the v1 stabilisation period.
    model_config = ConfigDict(extra="allow")

    anchor_id: str
    structure_type: str = Field(description="e.g. drain, point, cove, oyster_bar, island_edge")
    scale: str = Field(description="'major' or 'minor'")
    anchor_center_latlon: tuple[float, float]
    geometry: ObservedGeometry
    orientation_deg: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    source_images_used: list[str] = Field(default_factory=list)

    # --- Phase C v1 additions (TASK-4) ---
    state: StructureState = Field(default="draft")
    phase_history: list[PhaseEvent] = Field(default_factory=list)
    provenance: Provenance
    findings: list[Finding] = Field(default_factory=list)
    seed_z18_fetch_plan: Z18FetchPlan | None = Field(
        default=None,
        description="Set at PLAN_CAPTURE; None if coord-gen failed for this anchor",
    )
    priority_rank: int | None = Field(
        default=None,
        description="Position in the cell's anchor priority order (1 = highest)",
    )
    zone_id: str | None = Field(
        default=None,
        description="ID of the LocalComplex / zone this anchor belongs to (from v3 zones)",
    )


class MemberFeature(BaseModel):
    """A secondary visible feature that functions with the anchor. Observed."""

    name: str = Field(description="Human-readable label, e.g. 'left flanking point'")
    feature_type: str = Field(
        description="Constrained vocabulary: point | basin | bar | shoreline | "
        "channel | pocket | spit | mangrove_finger",
    )
    geometry: ObservedGeometry
    notes: str = ""


class LocalComplex(BaseModel):
    """Anchor + its member features. Members are stored individually,
    NOT unioned into a single polygon. An optional envelope may be included
    for visualization; when present it is always interpreted geometry."""

    complex_id: str
    anchor_id: str
    members: list[MemberFeature] = Field(default_factory=list)
    relationship_summary: str = ""
    envelope: InterpretedGeometry | None = Field(
        default=None,
        description="Optional outer boundary covering the complex; always interpreted",
    )


class InfluenceZone(BaseModel):
    """Area where the anchor still materially explains fishability. Interpreted."""

    influence_zone_id: str
    anchor_id: str
    geometry: InterpretedGeometry
    influence_shape_type: str = Field(
        default="radial",
        description="e.g. radial | directional | channelized | fan | funnel",
    )
    bounded_by: list[str] = Field(default_factory=list)
    dominance_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    competing_structures: list[str] = Field(default_factory=list)


class FishableSubzone(BaseModel):
    """A practical fishing target within the influence zone. Observed."""

    subzone_id: str
    anchor_id: str
    subzone_type: str = Field(
        description="v1 whitelist: drain_throat | point_tip | oyster_bar_edge | "
        "pocket_mouth | island_tip_seam",
    )
    geometry: ObservedGeometry
    relative_priority: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# --- Bookkeeping objects ---


class DeferredAnchor(BaseModel):
    """An anchor discovered but not processed; kept in the registry for a future run."""

    anchor_id: str
    structure_type: str
    scale: str
    confidence: float
    expected_relevance: float
    approx_bbox_px_z16: tuple[int, int, int, int]
    rationale: str = ""
    rank: float = 0.0


class OverlapEntry(BaseModel):
    """One entry in the registry's overlap_report."""

    anchor_id_a: str
    anchor_id_b: str
    level: str  # "anchor" | "complex_member" | "influence" | "subzone"
    iou: float
    policy: str  # "kept" | "subordinated"


class SegmentationIssue(BaseModel):
    """One entry recording a fallback or abnormality during extraction."""

    feature_id: str
    feature_level: str  # "anchor" | "complex_member" | "subzone"
    extractor_attempted: str
    fallback_used: str | None
    reason: str


class FailedIdentification(BaseModel):
    """One entry recording a feature dropped before extraction."""

    feature_id: str
    feature_level: str  # "anchor" | "complex_member" | "subzone"
    reason: str
    regeneration_attempted: bool


class StructurePhaseResult(BaseModel):
    """Full output of the structure phase for a single zoom-16 cell."""

    cell_id: str
    anchors: list[AnchorStructure] = Field(default_factory=list)
    complexes: list[LocalComplex] = Field(default_factory=list)
    influences: list[InfluenceZone] = Field(default_factory=list)
    subzones: list[FishableSubzone] = Field(default_factory=list)
    deferred: list[DeferredAnchor] = Field(default_factory=list)
    truncated_ids: list[str] = Field(default_factory=list)
    subordinated_ids: list[str] = Field(default_factory=list)
    overlap_report: list[OverlapEntry] = Field(default_factory=list)
    segmentation_issues: list[SegmentationIssue] = Field(default_factory=list)
    failed_identifications: list[FailedIdentification] = Field(default_factory=list)
    annotated_image_paths: dict[str, str] = Field(default_factory=dict)
    mosaic_image_paths: dict[str, str] = Field(default_factory=dict)
    registry_path: str = ""
    api_calls_used: int = 0
    tiles_fetched: int = 0
    truncated: bool = False
