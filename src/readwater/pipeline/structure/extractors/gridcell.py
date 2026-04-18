"""GridCellExtractor — turns a list of grid-cell labels into a pixel polygon.

Phase 1.5 primary extractor. Claude's IDENTIFY call returns cell labels
(e.g. ["C3", "C4", "D3"]); this extractor computes the bbox of the cell
union and returns it as a polygon. All four extraction modes route through
this extractor for now; the mode determines validation (min cell count),
not the polygon shape.

Phase 2 replaces this with a SAM-based extractor per mode, which takes the
cell bbox as a prompt and returns a feature-hugging precise polygon.
"""

from __future__ import annotations

from PIL import Image

from readwater.pipeline.structure.extractors.base import (
    ExtractorOutput,
    GeometryExtractor,
)
from readwater.pipeline.structure.grid_overlay import (
    cells_to_polygon,
)


class GridCellExtractor(GeometryExtractor):
    """Build a polygon from a cell-label list + grid metadata.

    This extractor ignores the image pixels entirely — the polygon is a
    pure function of the cell labels and the grid geometry. The caller
    passes the cell list via the `positive_points` parameter of the
    existing GeometryExtractor interface, encoded as [(row, col), ...]
    integer pairs. (Cell LABELS are parsed upstream; this extractor
    consumes already-parsed row/col integer pairs.)
    """

    def __init__(self, mode: str, grid_rows: int, grid_cols: int, image_size: tuple[int, int]):
        if mode not in ("region", "corridor", "point_feature", "edge_band"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self._rows = grid_rows
        self._cols = grid_cols
        self._image_size = image_size

    def extract(
        self,
        image: Image.Image,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> ExtractorOutput:
        # positive_points here are (row, col) pairs already parsed from labels.
        # Convert them back to label strings the way grid_overlay expects, or
        # inline the geometry directly. Inline is simpler.
        if not positive_points:
            return ExtractorOutput(
                pixel_polygon=[], confidence=None,
                extractor_name="gridcell",
            )

        labels = []
        from readwater.pipeline.structure.grid_overlay import row_label
        for row, col in positive_points:
            labels.append(f"{row_label(row)}{col + 1}")

        poly = cells_to_polygon(labels, self._rows, self._cols, self._image_size)
        if bbox_hint is not None:
            bx, by, bw, bh = bbox_hint
            x1 = max(bx, 0)
            y1 = max(by, 0)
            x2 = min(bx + bw, self._image_size[0])
            y2 = min(by + bh, self._image_size[1])
            poly = [
                (max(x1, min(x2, x)), max(y1, min(y2, y)))
                for (x, y) in poly
            ]

        return ExtractorOutput(
            pixel_polygon=poly,
            confidence=None,
            extractor_name="gridcell",
        )
