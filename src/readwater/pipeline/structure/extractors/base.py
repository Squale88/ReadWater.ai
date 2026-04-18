"""Abstract geometry extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class ExtractorOutput:
    """Result of running a GeometryExtractor on seed points."""

    pixel_polygon: list[tuple[int, int]]
    confidence: float | None
    extractor_name: str  # "clickbox" | "sam2_region" | "fallback" | ...


class GeometryExtractor(ABC):
    """Converts seed points into a pixel polygon on a given image.

    Extractors are stateless after construction; they are safe to reuse across
    many features. `mode` labels which extraction shape class the extractor
    produces ('region' | 'corridor' | 'point_feature' | 'edge_band').
    """

    mode: str  # subclasses set this

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> ExtractorOutput:
        """Build a polygon for the feature the positive points identify.

        Args:
            image: the source image; extractors may read pixel values here in
                Phase 2+ (Phase 1 extractors ignore it and use only geometry).
            positive_points: list of (x, y) pixel coordinates inside the feature.
            negative_points: list of (x, y) pixel coordinates nearby but outside.
                Used by Phase 2 extractors (SAM); Phase 1 ClickBox ignores them.
            bbox_hint: optional (x, y, w, h) constraining the extractor. If given,
                the returned polygon will be intersected with this rectangle.

        Returns:
            ExtractorOutput with the polygon and provenance.
        """
