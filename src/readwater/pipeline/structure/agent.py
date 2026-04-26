"""Phase C structure-phase entry point.

This module is the integration seam between the cell analyzer and the
Phase C v1 anchor discovery pipeline (`anchor_discovery.run_anchor_discovery`).

The legacy z16 grid-cell DISCOVER → IDENTIFY → EXTRACT pipeline that used
to live in this file was replaced by config-driven v3 + coord-gen + plan
capture. See `docs/PHASE_C_DISCOVERY_PIPELINE.md` for the new flow and
`docs/PHASE_C_TASKS.md` (TASK-6 + TASK-7) for the migration history.

Substructure work (LocalComplex, MemberFeature, FishableSubzone,
InfluenceZone) is Phase D — it consumes the AnchorStructure[] this module
emits and runs in a separate orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from readwater.api.providers.base import ImageProvider
from readwater.models.structure import StructurePhaseResult
from readwater.pipeline.structure.anchor_discovery import (
    AnchorDiscoveryConfig,
    AnchorDiscoveryInputs,
    run_anchor_discovery,
)

logger = logging.getLogger(__name__)


# --- State enum (documentary; the agent is now a thin wrapper) ---


class StructurePhase(str, Enum):
    DISCOVER = "discover"
    COORDS = "coords"
    PLAN_CAPTURE = "plan_capture"
    ASSEMBLE = "assemble"


# --- Bookkeeping (kept for source compatibility with cell_analyzer) ---


@dataclass
class StructureBudget:
    """Per-cell budget bookkeeping. The new flow consumes a fixed number of
    LLM calls (1-2 v3 + 1-2 coord-gen depending on comparison mode), so the
    detailed per-anchor sub-budgets that the legacy IDENTIFY/EXTRACT loop
    needed are no longer used here. Phase D maintains its own budget."""

    calls_per_anchor: int = 15            # legacy default; ignored by new flow
    tiles_per_anchor: int = 25            # legacy default; ignored by new flow
    max_anchors_per_cell: int = 3         # legacy default; ignored by new flow
    continuation_loop_cap: int = 2        # legacy default; ignored by new flow
    calls_used: int = 0
    tiles_used: int = 0


@dataclass
class StructurePaths:
    """Path layout for per-cell structure-phase artifacts."""

    cell_root: Path
    structures_root: Path
    discovery_image_path: Path
    registry_path: Path
    grid_overlay_dir: Path

    @classmethod
    def for_cell(cls, base_output_dir: Path, cell_id: str) -> "StructurePaths":
        cell_root = base_output_dir / "cells" / cell_id
        structures_root = cell_root / "structures"
        return cls(
            cell_root=cell_root,
            structures_root=structures_root,
            discovery_image_path=structures_root / "anchors_discovery.png",
            registry_path=structures_root / "registry.json",
            grid_overlay_dir=structures_root / "grid_overlays",
        )


# --- Public entry point ---


async def run_structure_phase(
    cell_id: str,
    cell_center: tuple[float, float],
    z15_image_path: str,
    z16_image_path: str,
    provider: ImageProvider,
    base_output_dir: Path,
    parent_context: str = "",
    coverage_miles: float = 0.37,
    budget: StructureBudget | None = None,
    extractor_registry: dict | None = None,  # ignored under v1 flow; Phase D uses it
    evidence_masks: dict[str, str] | None = None,  # superseded by discovery_config.inject_evidence
    discovery_config: AnchorDiscoveryConfig | None = None,
) -> StructurePhaseResult:
    """Run Phase C anchor discovery for one confirmed zoom-16 cell.

    Phase C v1 flow (replaces the legacy z16 grid-cell DISCOVER):
      DISCOVER (v3) -> COORDS (coord-gen) -> PLAN_CAPTURE -> ASSEMBLE
      → emits draft AnchorStructure[] with seed Z18FetchPlans.

    Substructure work (LocalComplex / InfluenceZone / FishableSubzone) is
    Phase D and runs downstream of this call against the emitted anchors.
    The legacy fields on `StructurePhaseResult` (complexes, influences,
    subzones, deferred, segmentation_issues, etc.) are left empty here;
    Phase D populates them.

    Args:
      cell_id, cell_center, z16_image_path: standard per-cell identifiers.
      z15_image_path, provider, parent_context, evidence_masks, budget,
        extractor_registry: kept on the signature for source compatibility
        with `cell_analyzer.run`; only `coverage_miles` is consumed by the
        new v1 flow. `evidence_masks` is superseded by
        `discovery_config.inject_evidence` which auto-discovers masks.
      discovery_config: AnchorDiscoveryConfig — pipeline mode/options. Default
        is `AnchorDiscoveryConfig()` (v3=nogrid, coords=grid, evidence off).
    """
    # Deferred to break the cycle: context_bundle imports from this package.
    from readwater.pipeline.context_bundle import bundle_path_for, load_bundle

    config = discovery_config or AnchorDiscoveryConfig()
    base_dir = Path(base_output_dir)
    cell_struct_dir = bundle_path_for(base_dir, cell_id).parent
    cell_struct_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = cell_struct_dir / "context_bundle.json"
    if not bundle_path.exists():
        logger.warning(
            "[structure:%s] context_bundle.json missing at %s — returning empty result. "
            "Run scripts/build_z16_bundles.py to produce one.",
            cell_id, bundle_path,
        )
        return StructurePhaseResult(cell_id=cell_id, truncated=True)
    bundle = load_bundle(bundle_path)

    z16_path = Path(z16_image_path)
    inputs = AnchorDiscoveryInputs(
        cell_id=cell_id,
        bundle=bundle,
        z16_image_path=z16_path,
        z16_grid_overlay_path=z16_path.parent / f"{z16_path.stem}_grid_8x8.png",
        overlay_z15_path=cell_struct_dir / "overlay_z15_same_center.png",
        overlay_z14_path=cell_struct_dir / "overlay_z14_parent.png",
        overlay_z12_path=cell_struct_dir / "overlay_z12_grandparent.png",
        z16_center_latlon=cell_center,
        coverage_miles=coverage_miles,
        # area_root = parent of "images" dir, e.g. data/areas/<area>
        area_root=base_dir.parent if base_dir.name == "images" else None,
    )

    logger.info(
        "[structure:%s] anchor_discovery v3=%s coords=%s evidence=%s",
        cell_id, config.v3_mode, config.coords_mode, config.inject_evidence,
    )
    result = await run_anchor_discovery(inputs, config, output_dir=cell_struct_dir)
    # Approximate API call count: 1 per LLM call. Comparison modes double up.
    api_calls = 0
    api_calls += 2 if config.v3_mode == "comparison" else 1
    api_calls += 2 if config.coords_mode == "comparison" else 1
    result.api_calls_used = api_calls
    return result
