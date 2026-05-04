"""Geometry extractors and dispatch — DEPRECATED.

Part of the LLM-driven structure pipeline replaced by ``readwater.pipeline.cv``.
See ``DEPRECATED.md`` at the repo root.

The agent chooses an extractor per feature based on (feature_type, extraction_mode).
Phase 1.5 default is GridCellExtractor (cells→bbox polygon). ClickBoxExtractor
stays available as a fallback. Phase 2 adds SAM-based extractors for the
region/corridor/edge_band modes without changing this module's API.
"""

from __future__ import annotations

from readwater.pipeline.structure.extractors.base import (
    ExtractorOutput,
    GeometryExtractor,
)
from readwater.pipeline.structure.extractors.clickbox import ClickBoxExtractor
from readwater.pipeline.structure.extractors.gridcell import GridCellExtractor
from readwater.pipeline.structure.extractors.modes import (
    STRUCTURE_TYPE_TO_MODE,
    SUBZONE_TYPE_TO_MODE,
    is_subzone_type_allowed,
    mode_for,
)


def build_gridcell_registry(
    grid_rows: int,
    grid_cols: int,
    image_size: tuple[int, int],
) -> dict[str, GeometryExtractor]:
    """Construct a per-mode GridCellExtractor registry for a specific image.

    The registry must be rebuilt per image because the extractor carries the
    grid dimensions and image size it was built for. Callers typically do this
    once per mosaic before running extraction.
    """
    return {
        "region": GridCellExtractor("region", grid_rows, grid_cols, image_size),
        "corridor": GridCellExtractor("corridor", grid_rows, grid_cols, image_size),
        "point_feature": GridCellExtractor("point_feature", grid_rows, grid_cols, image_size),
        "edge_band": GridCellExtractor("edge_band", grid_rows, grid_cols, image_size),
    }


# Fallback registry using ClickBoxExtractor — used when cell parsing fails.
_FALLBACK_REGISTRY: dict[str, GeometryExtractor] = {
    "region": ClickBoxExtractor(mode="region"),
    "corridor": ClickBoxExtractor(mode="corridor"),
    "point_feature": ClickBoxExtractor(mode="point_feature"),
    "edge_band": ClickBoxExtractor(mode="edge_band"),
}


def get_extractor(mode: str, registry: dict | None = None) -> GeometryExtractor:
    """Resolve a mode name to an extractor instance.

    `registry` is typically the per-image gridcell registry returned by
    `build_gridcell_registry`. Falls back to the clickbox fallback registry
    if `registry` is None or doesn't have the requested mode.
    """
    reg = registry if registry is not None else _FALLBACK_REGISTRY
    if mode in reg:
        return reg[mode]
    return reg.get("region", _FALLBACK_REGISTRY["region"])


def get_fallback_extractor(mode: str) -> GeometryExtractor:
    """Always returns a ClickBoxExtractor for the given mode (used on extraction
    failures — guarantees some geometry even when the primary path fails)."""
    return _FALLBACK_REGISTRY.get(mode, _FALLBACK_REGISTRY["region"])


__all__ = [
    "ClickBoxExtractor",
    "ExtractorOutput",
    "GeometryExtractor",
    "GridCellExtractor",
    "STRUCTURE_TYPE_TO_MODE",
    "SUBZONE_TYPE_TO_MODE",
    "build_gridcell_registry",
    "get_extractor",
    "get_fallback_extractor",
    "is_subzone_type_allowed",
    "mode_for",
]
