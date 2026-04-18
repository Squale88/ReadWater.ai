"""Geometry extractors and dispatch.

A single routing function `extractor_for(feature_type, feature_level)` resolves
to a (mode, extractor) pair. Phase 1 ships only ClickBoxExtractor for all four
modes; Phase 2+ can register additional extractors (SAM for region, centerline
for corridor, etc.) without touching the agent.
"""

from __future__ import annotations

from readwater.pipeline.structure.extractors.base import (
    ExtractorOutput,
    GeometryExtractor,
)
from readwater.pipeline.structure.extractors.clickbox import ClickBoxExtractor
from readwater.pipeline.structure.extractors.modes import (
    STRUCTURE_TYPE_TO_MODE,
    SUBZONE_TYPE_TO_MODE,
    is_subzone_type_allowed,
    mode_for,
)

# --- Default extractor registry ---
#
# Phase 1: every mode routes to ClickBoxExtractor (mode-parameterized).
# Phase 2 can replace entries here (e.g. region -> SAM2RegionExtractor).
# Lookup via `get_extractor(mode)`.
_DEFAULT_REGISTRY: dict[str, GeometryExtractor] = {
    "region": ClickBoxExtractor(mode="region"),
    "corridor": ClickBoxExtractor(mode="corridor"),
    "point_feature": ClickBoxExtractor(mode="point_feature"),
    "edge_band": ClickBoxExtractor(mode="edge_band"),
}


def get_extractor(mode: str, registry: dict | None = None) -> GeometryExtractor:
    """Resolve a mode name to an extractor instance.

    `registry` overrides the default registry (for tests or Phase 2+ wiring).
    Falls back to the region extractor if an unknown mode is requested.
    """
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    if mode in reg:
        return reg[mode]
    return reg["region"]


__all__ = [
    "ClickBoxExtractor",
    "ExtractorOutput",
    "GeometryExtractor",
    "STRUCTURE_TYPE_TO_MODE",
    "SUBZONE_TYPE_TO_MODE",
    "get_extractor",
    "is_subzone_type_allowed",
    "mode_for",
]
