"""ClickBoxExtractor — Phase 1 geometry extractor.

Produces rectangular polygons sized and oriented from seed points. Four modes:
  - region:        oriented bbox around positives, padded
  - corridor:      thin rectangle from first positive to last, fixed width
  - point_feature: small square centered on the lone positive point
  - edge_band:     thin rectangle along the line defined by positives

Intentionally simple. The purpose is to ship a working Phase 1 with correct
architecture, seed-point plumbing, and provenance tracking. Phase 2 replaces
specific modes (region first) with more sophisticated extractors without
changing this module.

All padding constants live at the top of this file. They are expressed as
fractions of the image's shorter dimension so the extractor produces shapes
of consistent real-world size regardless of mosaic resolution.
"""

from __future__ import annotations

import math

from PIL import Image

from readwater.pipeline.structure.extractors.base import (
    ExtractorOutput,
    GeometryExtractor,
)

# Tuning constants (fractions of image shorter dim).
REGION_PAD_FRAC = 0.05
CORRIDOR_WIDTH_FRAC = 0.03
POINT_SIDE_FRAC = 0.04
EDGE_BAND_WIDTH_FRAC = 0.02


class ClickBoxExtractor(GeometryExtractor):
    """Rectangular-polygon extractor for all four modes.

    Usage: instantiate once per mode; the same instance is safe to reuse.
    """

    def __init__(
        self,
        mode: str,
        *,
        region_pad_frac: float = REGION_PAD_FRAC,
        corridor_width_frac: float = CORRIDOR_WIDTH_FRAC,
        point_side_frac: float = POINT_SIDE_FRAC,
        edge_band_width_frac: float = EDGE_BAND_WIDTH_FRAC,
    ):
        if mode not in ("region", "corridor", "point_feature", "edge_band"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self._region_pad = region_pad_frac
        self._corridor_width = corridor_width_frac
        self._point_side = point_side_frac
        self._edge_band_width = edge_band_width_frac

    def extract(
        self,
        image: Image.Image,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]],
        bbox_hint: tuple[int, int, int, int] | None = None,
    ) -> ExtractorOutput:
        w, h = image.size
        short = min(w, h)

        if not positive_points:
            return ExtractorOutput(pixel_polygon=[], confidence=None,
                                   extractor_name="clickbox")

        if self.mode == "region":
            poly = self._region(positive_points, short)
        elif self.mode == "corridor":
            poly = self._corridor(positive_points, short)
        elif self.mode == "point_feature":
            poly = self._point_feature(positive_points[0], short)
        else:  # edge_band
            poly = self._edge_band(positive_points, short)

        poly = _clip_polygon_to_image(poly, w, h)
        if bbox_hint is not None:
            poly = _intersect_with_bbox(poly, bbox_hint, w, h)
        return ExtractorOutput(
            pixel_polygon=poly,
            confidence=None,
            extractor_name="clickbox",
        )

    # --- mode implementations ---

    def _region(
        self, positives: list[tuple[int, int]], short: int,
    ) -> list[tuple[int, int]]:
        pad = int(self._region_pad * short)
        xs = [p[0] for p in positives]
        ys = [p[1] for p in positives]
        x0 = min(xs) - pad
        x1 = max(xs) + pad
        y0 = min(ys) - pad
        y1 = max(ys) + pad
        return _rect(x0, y0, x1, y1)

    def _corridor(
        self, positives: list[tuple[int, int]], short: int,
    ) -> list[tuple[int, int]]:
        width = int(self._corridor_width * short)
        if len(positives) == 1:
            # Fall through to point_feature behavior.
            return self._point_feature(positives[0], short)
        # Use first and last positives as the corridor endpoints; middle
        # positives are implicit "path hints" that ClickBox doesn't use yet.
        a = positives[0]
        b = positives[-1]
        return _thick_line_polygon(a, b, width)

    def _point_feature(
        self, point: tuple[int, int], short: int,
    ) -> list[tuple[int, int]]:
        side = int(self._point_side * short)
        half = side // 2
        x, y = point
        return _rect(x - half, y - half, x + half, y + half)

    def _edge_band(
        self, positives: list[tuple[int, int]], short: int,
    ) -> list[tuple[int, int]]:
        if len(positives) < 2:
            # Degenerate. Fall back to point_feature; agent will log as a
            # minor fallback because an edge_band needs a direction.
            return self._point_feature(positives[0], short)
        width = int(self._edge_band_width * short)
        a = positives[0]
        b = positives[-1]
        return _thick_line_polygon(a, b, width)


# --- shape helpers ---


def _rect(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """4-vertex rectangle polygon, CCW from top-left."""
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _thick_line_polygon(
    a: tuple[int, int], b: tuple[int, int], width: int,
) -> list[tuple[int, int]]:
    """A rectangle tracing the line from a to b with the given perpendicular width."""
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        # Degenerate; draw a small square at a.
        half = max(width // 2, 1)
        return _rect(ax - half, ay - half, ax + half, ay + half)
    # Unit perpendicular vector.
    ux = -dy / length
    uy = dx / length
    hw = width / 2.0
    p1 = (int(round(ax + ux * hw)), int(round(ay + uy * hw)))
    p2 = (int(round(bx + ux * hw)), int(round(by + uy * hw)))
    p3 = (int(round(bx - ux * hw)), int(round(by - uy * hw)))
    p4 = (int(round(ax - ux * hw)), int(round(ay - uy * hw)))
    return [p1, p2, p3, p4]


def _clip_polygon_to_image(
    polygon: list[tuple[int, int]], width: int, height: int,
) -> list[tuple[int, int]]:
    """Clamp each vertex to image bounds. Does not re-clip by Sutherland-Hodgman
    because ClickBox polygons are convex and typically mostly in-bounds; a
    vertex clamp is sufficient and preserves simple rectangular topology."""
    if not polygon:
        return []
    out: list[tuple[int, int]] = []
    for x, y in polygon:
        cx = max(0, min(width - 1, x))
        cy = max(0, min(height - 1, y))
        out.append((cx, cy))
    return out


def _intersect_with_bbox(
    polygon: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Clamp each polygon vertex to a bbox (x, y, w, h)."""
    if not polygon:
        return []
    bx, by, bw, bh = bbox
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(width - 1, bx + bw)
    y2 = min(height - 1, by + bh)
    out: list[tuple[int, int]] = []
    for x, y in polygon:
        cx = max(x1, min(x2, x))
        cy = max(y1, min(y2, y))
        out.append((cx, cy))
    return out
