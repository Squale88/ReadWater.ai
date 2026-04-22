"""Data models for the ReadWater area knowledge system."""

from readwater.models.area_knowledge import AreaKnowledge
from readwater.models.cell import Cell, CellAnalysis, CellScore
from readwater.models.context import (
    CandidateFeatureThread,
    CellContext,
    DirectObservation,
    EvidenceSummary,
    LineageRef,
    MorphologyInference,
    UnresolvedQuestion,
    VisualContextRef,
    VisualRole,
    Z16ContextBundle,
)

__all__ = [
    "AreaKnowledge",
    "CandidateFeatureThread",
    "Cell",
    "CellAnalysis",
    "CellContext",
    "CellScore",
    "DirectObservation",
    "EvidenceSummary",
    "LineageRef",
    "MorphologyInference",
    "UnresolvedQuestion",
    "VisualContextRef",
    "VisualRole",
    "Z16ContextBundle",
]

# CellAnalysis.context uses a forward reference to CellContext to avoid a
# circular import (context.py imports BoundingBox from cell.py). Now that
# both modules are loaded, resolve the forward ref.
CellAnalysis.model_rebuild()
