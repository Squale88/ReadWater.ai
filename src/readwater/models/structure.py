"""Structure-phase zone models.

Four levels of output from the structure-phase agent:
    AnchorStructure  - the main visible organizing feature
    LocalComplex     - nearby visible features that function with the anchor
    InfluenceZone    - area where the anchor still materially explains fishability
    FishableSubzone  - practical fishing targets within the complex/influence

All geometry is stored as both a pixel polygon (relative to a named source
image) and a lat/lon polygon (for future mapping).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneGeometry(BaseModel):
    """A polygon zone recorded in both pixel and lat/lon space.

    pixel_polygon is (x, y) in the coordinate frame of `image_ref` (e.g. the
    stitched mosaic). latlon_polygon is the same polygon reprojected to
    geographic coordinates. Both lists must have the same length and describe
    the same vertices in the same order.
    """

    pixel_polygon: list[tuple[int, int]] = Field(
        description="Polygon vertices in pixel space of the source image",
    )
    latlon_polygon: list[tuple[float, float]] = Field(
        description="Polygon vertices in (lat, lon), same order as pixel_polygon",
    )
    image_ref: str = Field(description="Name of the source image (e.g. 'mosaic', 'z16_cell')")


class AnchorStructure(BaseModel):
    """The main visible organizing feature of a fishing picture."""

    anchor_id: str
    structure_type: str = Field(description="e.g. drain, point, cove, oyster_bar, island_edge")
    scale: str = Field(description="'major' or 'minor'")
    anchor_center_latlon: tuple[float, float]
    geometry: ZoneGeometry
    orientation_deg: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_images_used: list[str] = Field(default_factory=list)
    rationale: str = ""


class LocalComplex(BaseModel):
    """Nearby visible features that function together with the anchor."""

    complex_id: str
    anchor_id: str
    member_features: list[str] = Field(default_factory=list)
    relationship_summary: str = ""
    geometry: ZoneGeometry
    context_dependencies: list[str] = Field(default_factory=list)


class InfluenceZone(BaseModel):
    """Area where the anchor still materially explains fishability.

    May extend beyond the visible anchor, but must not cover the entire image.
    """

    influence_zone_id: str
    anchor_id: str
    geometry: ZoneGeometry
    influence_shape_type: str = Field(
        default="radial",
        description="e.g. radial, directional, channelized, funnel",
    )
    bounded_by: list[str] = Field(default_factory=list)
    dominance_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    competing_structures: list[str] = Field(default_factory=list)


class FishableSubzone(BaseModel):
    """A practical fishing target within the influence zone."""

    subzone_id: str
    anchor_id: str
    subzone_type: str = Field(
        description="e.g. drain_throat, left_ambush_point, oyster_bar_edge",
    )
    geometry: ZoneGeometry
    relative_priority: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DeferredAnchor(BaseModel):
    """An anchor that was discovered but not processed through the full phase.

    Kept verbatim in the registry so a future run can resume it.
    """

    anchor_id: str
    structure_type: str
    scale: str
    confidence: float
    expected_relevance: float
    approx_bbox_px_z16: tuple[int, int, int, int]
    rationale: str = ""
    rank: float = 0.0


class OverlapEntry(BaseModel):
    """A single entry in the registry's overlap_report."""

    anchor_id_a: str
    anchor_id_b: str
    level: str  # "anchor" | "complex" | "influence" | "subzone"
    iou: float
    policy: str  # "kept" | "subordinated"


class StructurePhaseResult(BaseModel):
    """Full output of the structure phase for a single zoom-16 cell."""

    cell_id: str
    anchors: list[AnchorStructure] = Field(default_factory=list)
    complexes: list[LocalComplex] = Field(default_factory=list)
    influences: list[InfluenceZone] = Field(default_factory=list)
    subzones: list[FishableSubzone] = Field(default_factory=list)
    deferred: list[DeferredAnchor] = Field(default_factory=list)
    truncated_ids: list[str] = Field(default_factory=list)
    failed_geometry_ids: list[str] = Field(default_factory=list)
    subordinated_ids: list[str] = Field(default_factory=list)
    overlap_report: list[OverlapEntry] = Field(default_factory=list)
    annotated_image_paths: dict[str, str] = Field(default_factory=dict)
    mosaic_image_paths: dict[str, str] = Field(default_factory=dict)
    registry_path: str = ""
    api_calls_used: int = 0
    tiles_fetched: int = 0
    truncated: bool = False
