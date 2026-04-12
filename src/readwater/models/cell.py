"""Cell models representing units of the recursive satellite analysis grid."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ZoomLevel(int, Enum):
    """Named zoom levels for the recursive analysis pipeline."""

    MACRO = 0       # ~30 miles — macro hydrology, major water bodies
    ZONE = 1        # ~10 miles — distinct fishing zones
    SUBZONE = 2     # ~3.3 miles — sub-zones within a zone
    AREA = 3        # ~1.1 miles — individual fishing areas
    STRUCTURE = 4   # ~0.37 miles — structure-level detail


class BoundingBox(BaseModel):
    """Geographic bounding box defined by corner coordinates."""

    north: float = Field(description="Northern latitude boundary")
    south: float = Field(description="Southern latitude boundary")
    east: float = Field(description="Eastern longitude boundary")
    west: float = Field(description="Western longitude boundary")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.north + self.south) / 2, (self.east + self.west) / 2)


class CellScore(BaseModel):
    """Fishing potential score for a single sub-cell within a grid analysis."""

    row: int = Field(ge=0, description="Row index in the parent grid (0-indexed)")
    col: int = Field(ge=0, description="Column index in the parent grid (0-indexed)")
    score: float = Field(ge=0, le=10, description="Fishing potential score (0=no interest, 10=prime)")
    summary: str = Field(description="Brief description of what was observed in this sub-cell")
    center: tuple[float, float] = Field(description="(lat, lon) center of this sub-cell")


class CellAnalysis(BaseModel):
    """Claude's analysis result for a single cell's satellite image."""

    overall_summary: str = Field(description="High-level description of the cell's fishing character")
    sub_scores: list[CellScore] = Field(description="Scores for each sub-cell in the grid")
    structure_types: list[str] = Field(
        default_factory=list,
        description="Types of fishable structure identified (e.g., grass flat, oyster bar, channel)",
    )
    model_used: str = Field(default="claude-sonnet-4-20250514")
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)


class Cell(BaseModel):
    """A single cell in the recursive area knowledge tree.

    Each cell represents a square geographic region analyzed via satellite imagery.
    Cells form a tree: the root is the full area (~30 miles), and children are
    sub-cells that scored above the pruning threshold.
    """

    id: str = Field(description="Unique cell identifier (e.g., 'root', 'root-1-2', 'root-1-2-0-1')")
    parent_id: str | None = Field(default=None, description="Parent cell ID, None for root")
    center: tuple[float, float] = Field(description="(lat, lon) center point")
    size_miles: float = Field(description="Side length of the square cell in miles")
    zoom_level: int = Field(ge=0, description="Depth in the recursion tree (0=root)")
    bbox: BoundingBox = Field(description="Geographic bounding box")
    analysis: CellAnalysis | None = Field(default=None, description="Claude's analysis, None if not yet analyzed")
    image_path: str | None = Field(default=None, description="Path to saved satellite image")
    children_ids: list[str] = Field(default_factory=list, description="IDs of child cells that passed threshold")
