"""Area knowledge model — the top-level container for a fully analyzed fishing zone."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from readwater.models.cell import Cell


class AreaKnowledge(BaseModel):
    """Complete area knowledge for a coastal fishing zone.

    Built by the recursive cell analyzer pipeline. Contains the full tree of
    analyzed cells from macro (~30 mi) down to structure-level (~0.37 mi) detail.
    This is the primary artifact consumed by the Forecast Engine.
    """

    name: str = Field(description="Human-readable area name (e.g., 'Tampa Bay')")
    root_center: tuple[float, float] = Field(description="(lat, lon) center of the root analysis cell")
    root_size_miles: float = Field(description="Side length of the root cell in miles")
    cells: dict[str, Cell] = Field(default_factory=dict, description="All cells keyed by cell ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    total_cells_analyzed: int = Field(default=0)
    total_cells_pruned: int = Field(default=0)

    def add_cell(self, cell: Cell) -> None:
        """Add an analyzed cell to the knowledge tree."""
        self.cells[cell.id] = cell
        self.total_cells_analyzed += 1
        self.updated_at = datetime.utcnow()

    def get_children(self, cell_id: str) -> list[Cell]:
        """Get all direct children of a cell."""
        cell = self.cells.get(cell_id)
        if not cell:
            return []
        return [self.cells[cid] for cid in cell.children_ids if cid in self.cells]

    def get_cells_at_level(self, zoom_level: int) -> list[Cell]:
        """Get all cells at a specific Google Maps zoom level."""
        return [c for c in self.cells.values() if c.zoom_level == zoom_level]

    def get_cells_at_depth(self, depth: int) -> list[Cell]:
        """Get all cells at a specific tree depth (0=root)."""
        return [c for c in self.cells.values() if c.depth == depth]

    def get_leaf_cells(self) -> list[Cell]:
        """Get all leaf cells (no children) — these are the finest-grained analysis."""
        return [c for c in self.cells.values() if not c.children_ids]
