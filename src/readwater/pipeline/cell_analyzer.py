"""Recursive cell analyzer — the core of the area knowledge pipeline.

This module implements multi-scale satellite imagery analysis that builds
structured area knowledge by recursively analyzing coastal zones at increasing
zoom levels (10 → 12 → 14 → 16 → 18), using 4x4 grids aligned with
Google Maps tile math (2 zoom levels = 4x linear = 16 sub-cells).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from readwater.api.providers.registry import ImageProviderRegistry
from readwater.models.cell import BoundingBox, Cell, CellAnalysis, CellScore
from readwater.models.context import CellContext, LineageRef
from readwater.pipeline.context_bundle import (
    assemble_z16_bundle,
    bundle_path_for,
    persist_bundle,
)
from readwater.pipeline.cv.cell_pipeline import run_cell_full
from readwater.pipeline.cv.discovery import evaluate_subcells

# Tag stored in CellAnalysis.model_used to mark records produced by the
# deterministic CV rubric (vs the LLM-driven scoring it replaces).
CV_RUBRIC_VERSION = "cv-rubric-v1"

logger = logging.getLogger(__name__)

# --- Constants ---

MILES_PER_DEG_LAT = 69.0
EARTH_CIRCUMFERENCE_MILES = 24901.0
SECTIONS = 4
TERMINAL_ZOOM = 16  # CV pipeline owns zoom 18 and below
STRUCTURE_PHASE_ZOOM = 16
Z15_CONTEXT_PROVIDER_ROLE = "overview"
VALID_START_ZOOMS = {10, 12}


# --- Coordinate geometry ---


def _miles_per_deg_lon(lat: float) -> float:
    """Miles per degree of longitude at a given latitude."""
    return MILES_PER_DEG_LAT * math.cos(math.radians(lat))


def ground_coverage_miles(zoom: int, lat: float, image_size: int = 640) -> float:
    """Approximate ground coverage in miles for a Google Maps image.

    At zoom Z, one 256px tile covers (earth_circumference * cos(lat)) / 2^Z miles.
    A 640px image spans 2.5 tiles.
    """
    tiles = image_size / 256
    return tiles * EARTH_CIRCUMFERENCE_MILES * math.cos(math.radians(lat)) / (2**zoom)


def _make_bbox(center: tuple[float, float], size_miles: float) -> BoundingBox:
    """Calculate a bounding box from a center point and side length in miles."""
    lat, lon = center
    half_lat = (size_miles / 2) / MILES_PER_DEG_LAT
    half_lon = (size_miles / 2) / _miles_per_deg_lon(lat)
    return BoundingBox(
        north=lat + half_lat,
        south=lat - half_lat,
        east=lon + half_lon,
        west=lon - half_lon,
    )


def _subdivide_bbox(
    bbox: BoundingBox, sections: int,
) -> list[tuple[int, int, tuple[float, float]]]:
    """Divide a bounding box into a sections x sections grid.

    Returns a list of (row, col, (center_lat, center_lon)) tuples.
    Row 0 is the northernmost row, col 0 is the westernmost column.
    """
    cell_height = (bbox.north - bbox.south) / sections
    cell_width = (bbox.east - bbox.west) / sections

    sub_cells = []
    for row in range(sections):
        for col in range(sections):
            center_lat = bbox.north - (row + 0.5) * cell_height
            center_lon = bbox.west + (col + 0.5) * cell_width
            sub_cells.append((row, col, (center_lat, center_lon)))
    return sub_cells


def _sub_cell_bbox(
    parent_bbox: BoundingBox, row: int, col: int, sections: int,
) -> BoundingBox:
    """Calculate the bounding box of a single sub-cell within a parent grid."""
    cell_height = (parent_bbox.north - parent_bbox.south) / sections
    cell_width = (parent_bbox.east - parent_bbox.west) / sections
    return BoundingBox(
        north=parent_bbox.north - row * cell_height,
        south=parent_bbox.north - (row + 1) * cell_height,
        east=parent_bbox.west + (col + 1) * cell_width,
        west=parent_bbox.west + col * cell_width,
    )


# --- Tree identifiers and filenames ---


def _make_cell_id(parent_id: str | None, row: int, col: int) -> str:
    """Generate a cell ID using 1-based cell numbers matching the grid overlay.

    Cell numbers run 1-16, left to right, top to bottom in a 4x4 grid:
        row 0: cells 1-4,  row 1: cells 5-8,
        row 2: cells 9-12, row 3: cells 13-16
    """
    if parent_id is None:
        return "root"
    cell_num = row * SECTIONS + col + 1
    return f"{parent_id}-{cell_num}"


def _image_filename(cell_id: str, depth: int, provider_name: str | None = None) -> str:
    """Generate the image filename from a cell's tree path.

    Filenames use the cell number chain so they match the grid overlay directly:
        "root"       -> "z0.png"
        "root-14"    -> "z0_14.png"        (cell 14 of root)
        "root-14-3"  -> "z0_14_3.png"      (cell 3 of cell 14)
        "root-14-3"  -> "z0_14_3_naip.png" (with provider suffix)
    """
    if cell_id == "root":
        base = "z0"
    else:
        nums = cell_id.removeprefix("root-").replace("-", "_")
        base = f"z0_{nums}"

    if provider_name:
        return f"{base}_{provider_name}.png"
    return f"{base}.png"


def _role_for_zoom(zoom: int) -> str:
    """Determine the provider role for a given Google Maps zoom level.

    Zoom 16 uses structure-role providers (NAIP etc. where available); zooms
    above 16 are owned by the structure phase itself and fetched via the same
    structure provider.
    """
    if zoom >= 16:
        return "structure"
    return "overview"


# --- Deterministic CV-rubric scoring (replaces LLM dual_pass_grid_scoring) ---


def _score_subcells_via_cv_rubric(
    bbox: BoundingBox,
    parent_zoom: int,
    sub_cells_map: dict[tuple[int, int], tuple[float, float]],
    output_dir: Path,
) -> CellAnalysis:
    """Score a cell's 16 sub-cells using the deterministic CV rubric.

    Wraps ``readwater.pipeline.cv.discovery.evaluate_subcells``: fetches
    detail (parent_zoom) + isolation (parent_zoom-1) styled water tiles
    centered on this cell, computes the per-sub-cell metrics
    (water_pct, land_pct, widest_water_m, n_ccs), applies the rubric,
    and projects the binary keep/drop decision into the legacy CellScore
    shape (5.0 = keep, 0.0 = drop) so the rest of cell_analyzer's
    recursion path is unchanged.

    Tiles are cached under ``output_dir / "_discovery_cache"`` and reused
    across runs / sibling cells that share an evaluator center.

    The ``summary`` field on each CellScore captures the rubric metrics
    (e.g. ``"water_pct=12.3%, land_pct=78.1%, widest=240m"``) plus the
    drop reason for skipped cells, so downstream consumers can audit
    every decision without re-running the rubric.
    """
    parent_bb = {
        "north": bbox.north, "south": bbox.south,
        "east":  bbox.east,  "west":  bbox.west,
    }
    cache_dir = Path(output_dir) / "_discovery_cache"
    # Label used for tile filenames; coords-based so sibling cells at the same
    # zoom don't collide and so re-runs of the same cell hit the cache.
    label = (f"z{parent_zoom}"
             f"_{round(bbox.south, 4)}_{round(bbox.west, 4)}")
    evals = evaluate_subcells(parent_bb, parent_zoom, cache_dir, label)

    sub_scores: list[CellScore] = []
    for ev in evals:
        row = (ev.cell_num - 1) // SECTIONS
        col = (ev.cell_num - 1) % SECTIONS
        # Each sub-cell's geographic center comes from the geometric
        # subdivision the caller already computed (sub_cells_map keyed by
        # row/col -> (lat, lon)).
        sub_center = sub_cells_map.get((row, col), bbox_center_tuple(bbox))
        score = 5.0 if ev.kept else 0.0
        summary_parts = [
            f"water_pct={ev.water_pct}%",
            f"land_pct={ev.land_pct}%",
            f"widest={ev.widest_m}m",
            f"ccs={ev.n_ccs}",
        ]
        if ev.drop_reason:
            summary_parts.append(f"DROP={ev.drop_reason}")
        sub_scores.append(CellScore(
            row=row, col=col,
            score=score,
            summary="; ".join(summary_parts),
            center=sub_center,
        ))

    n_kept = sum(1 for s in sub_scores if s.score >= 5.0)
    return CellAnalysis(
        overall_summary=(
            f"Deterministic CV rubric (z{parent_zoom} detail + "
            f"z{parent_zoom-1} isolation): "
            f"{n_kept}/{len(sub_scores)} sub-cells kept"
        ),
        sub_scores=sub_scores,
        hydrology_notes="",
        model_used=CV_RUBRIC_VERSION,
    )


def bbox_center_tuple(bbox: BoundingBox) -> tuple[float, float]:
    """Helper: BoundingBox -> (lat, lon) center as a tuple."""
    return ((bbox.north + bbox.south) / 2, (bbox.east + bbox.west) / 2)


# --- Run state ---


@dataclass
class _RunState:
    """Mutable state shared across all recursive calls in a single run."""

    api_calls: int = 0
    max_api_calls: int = 50
    dry_run: bool = False
    output_dir: Path = field(default_factory=lambda: Path("data/areas/default/images"))
    metadata: list[dict] = field(default_factory=list)
    registry: ImageProviderRegistry | None = None
    threshold: float = 4.0
    start_zoom: int = 10


def _save_metadata(state: _RunState) -> None:
    """Write metadata.json alongside the images."""
    if not state.metadata:
        return
    state.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = state.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(state.metadata, indent=2))
    logger.info("Saved metadata for %d cells to %s", len(state.metadata), meta_path)


# --- Public entry point ---


async def analyze_cell(
    center: tuple[float, float],
    registry: ImageProviderRegistry,
    start_zoom: int = 10,
    threshold: float = 4.0,
    parent_context: str = "",
    max_api_calls: int = 50,
    max_depth: int = 2,
    dry_run: bool = False,
    area_name: str = "default",
    output_dir: str | None = None,
) -> list[Cell]:
    """Recursively analyze a geographic cell via satellite imagery.

    Builds a tree of cells by fetching satellite images at increasing zoom
    levels (stepping by 2 per depth) and scoring sub-cells on a 4x4 grid.

    Args:
        center: (lat, lon) center point of the root cell.
        registry: Provider registry mapping roles to image providers.
        start_zoom: Starting Google Maps zoom level (10 or 12).
        threshold: Minimum score (0-10) to recurse into a sub-cell.
        parent_context: Summary from parent cell (for future Claude analysis).
        max_api_calls: Hard cap on API calls per run.
        max_depth: Max recursion depth (0=root only).
        dry_run: If True, skip API calls and log what would be fetched.
        area_name: Name for the output directory.
        output_dir: Override output path.

    Returns:
        Flat list of all Cell objects in the analysis tree.
    """
    if start_zoom not in VALID_START_ZOOMS:
        raise ValueError(
            f"start_zoom must be one of {sorted(VALID_START_ZOOMS)}, got {start_zoom}"
        )

    out = Path(output_dir) if output_dir else Path(f"data/areas/{area_name}/images")
    state = _RunState(
        max_api_calls=max_api_calls,
        dry_run=dry_run,
        output_dir=out,
        registry=registry,
        threshold=threshold,
        start_zoom=start_zoom,
    )

    cells = await _analyze_recursive(
        center=center,
        zoom=start_zoom,
        depth=0,
        cell_id="root",
        parent_cell_id=None,
        parent_context=parent_context,
        max_depth=max_depth,
        state=state,
    )

    _save_metadata(state)
    return cells


# --- Recursive implementation ---


async def _analyze_recursive(
    center: tuple[float, float],
    zoom: int,
    depth: int,
    cell_id: str,
    parent_cell_id: str | None,
    parent_context: str,
    max_depth: int,
    state: _RunState,
    bbox: BoundingBox | None = None,
    ancestor_lineage: list[LineageRef] | None = None,
    ancestor_contexts: dict[str, CellContext] | None = None,
    position_in_parent: tuple[int, int] | None = None,
) -> list[Cell]:
    """Internal recursive implementation of cell analysis.

    Step 7 additions (data flow only; no LLM invocation here):
      - ancestor_lineage: ordered LineageRef list from root toward the parent,
        excluding self. Children receive the parent's chain extended with
        the parent's own LineageRef.
      - ancestor_contexts: cell_id -> CellContext snapshots produced by
        Step 8's descriptive pass on each retained cell. This step carries
        the dict through unchanged; population happens in Step 8.
      - position_in_parent: (row, col) of this cell within the parent's
        4x4 grid. None for the root.
    """
    if ancestor_lineage is None:
        ancestor_lineage = []
    if ancestor_contexts is None:
        ancestor_contexts = {}

    size_miles = ground_coverage_miles(zoom, center[0])
    if bbox is None:
        bbox = _make_bbox(center, size_miles)
    filename = _image_filename(cell_id, depth)
    provider_images: dict[str, str] = {}

    # --- Image procurement ---
    role = _role_for_zoom(zoom)

    if state.dry_run:
        if role == "structure" and state.registry:
            providers = state.registry.get_providers("structure")
            for p in providers:
                fname = _image_filename(cell_id, depth, p.name)
                logger.info("DRY RUN: would fetch %s from %s (zoom %d)", fname, p.name, zoom)
        else:
            logger.info("DRY RUN: would fetch %s (zoom %d, %.1f mi)", filename, zoom, size_miles)
    else:
        if role == "structure" and state.registry:
            providers = state.registry.get_providers("structure")
            for p in providers:
                if state.api_calls >= state.max_api_calls:
                    break
                if not p.supports_zoom(zoom):
                    continue
                if state.api_calls > 0:
                    await asyncio.sleep(0.5)
                fname = _image_filename(cell_id, depth, p.name)
                out_path = str(state.output_dir / fname)
                await p.fetch(center, zoom, out_path)
                state.api_calls += 1
                provider_images[p.name] = out_path
        else:
            if state.api_calls >= state.max_api_calls:
                logger.warning("API limit reached (%d), skipping cell %s", state.max_api_calls, cell_id)
                return []
            if state.api_calls > 0:
                await asyncio.sleep(0.5)
            provider = state.registry.get_default_provider("overview")
            out_path = str(state.output_dir / filename)
            await provider.fetch(center, zoom, out_path)
            state.api_calls += 1
            provider_images[provider.name] = out_path

        if not provider_images:
            return []

    # --- Analyze image ---
    sub_cells_geo = _subdivide_bbox(bbox, SECTIONS)
    sub_cells_map = {(r, c): ctr for r, c, ctr in sub_cells_geo}

    # Step 9: z16 bundle path is only set in the live branch below; keep a
    # None sentinel here so the metadata block downstream can reference it
    # unconditionally whether we ran dry or live.
    z16_bundle_path: str | None = None

    if state.dry_run:
        analysis = CellAnalysis(
            overall_summary="Placeholder analysis (dry run)",
            sub_scores=[
                CellScore(row=r, col=c, score=5.0, summary="Placeholder", center=ctr)
                for r, c, ctr in sub_cells_geo
            ],
            model_used="placeholder",
        )
    else:
        first_image = next(iter(provider_images.values()))
        is_structure_phase_cell = zoom == STRUCTURE_PHASE_ZOOM

        # --- Deterministic CV-rubric scoring (replaces LLM dual-pass scoring) ---
        # See readwater.pipeline.cv.discovery.evaluate_subcells for the rubric:
        #   detail tile at parent_zoom + isolation tile at parent_zoom-1
        #   ocean_connected = perim_connected(detail) AND NOT isolated(isolation)
        #   per sub-cell: KEEP if land_pct >= 3% AND water_pct >= threshold
        #                 (level-2 only) AND widest_water_m >= 50m
        # Tile fetches happen inside evaluate_subcells via storage helpers; no
        # need to fetch the LLM-style grid overlay or wider context image.
        try:
            analysis = _score_subcells_via_cv_rubric(
                bbox, zoom, sub_cells_map, state.output_dir,
            )
        except Exception as e:  # noqa: BLE001 — capture any rubric failure
            logger.warning("CV rubric scoring failed for cell %s: %s", cell_id, e)
            analysis = CellAnalysis(
                overall_summary=f"CV rubric failed: {e}",
                sub_scores=[
                    CellScore(row=r, col=c, score=0.0, summary="rubric error", center=ctr)
                    for r, c, ctr in sub_cells_geo
                ],
                model_used=CV_RUBRIC_VERSION,
            )

        # --- Z16 context bundle assembly (Step 9) — SKELETON FORM ---
        # The bundle WAS the input package for the LLM-driven structure
        # pipeline (deleted in PR #19). The future analyzer that
        # determines wind/current/weather/season impact at the closest
        # zoom level will need *something* like this — but its exact
        # shape isn't pinned down yet (LLM vs. deterministic, what
        # inputs it needs).
        #
        # We keep ``assemble_z16_bundle`` running here as a structural
        # framework so the file shape and writer path stay alive for
        # future extension. We DO NOT pay for the expensive per-cell
        # inputs:
        #   * ``build_cell_context`` (LLM call per retained cell) — REMOVED
        #   * z15 same-center fetch (1 API call per z16 cell) — REMOVED
        # As a result the bundle's ``contexts`` dict is empty and its
        # ``visuals`` set is {Z16_LOCAL, Z14_PARENT, Z12_GRANDPARENT}
        # (no Z15_SAME_CENTER). Each visual references images already
        # on disk from the recursive image fetch; the bundle assembly
        # itself is essentially free.
        if is_structure_phase_cell and state.registry is not None:
            try:
                self_lineage_ref = LineageRef(
                    cell_id=cell_id,
                    zoom=zoom,
                    depth=depth,
                    center=center,
                    bbox=bbox,
                    image_path=first_image,
                    position_in_parent=position_in_parent,
                )
                bundle = assemble_z16_bundle(
                    self_lineage=self_lineage_ref,
                    self_context=None,
                    ancestor_lineage=ancestor_lineage,
                    ancestor_contexts={},
                    z15_same_center_path=None,
                    base_output_dir=state.output_dir,
                )
                z16_bundle_path = persist_bundle(
                    bundle,
                    bundle_path_for(state.output_dir, cell_id),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Z16 bundle assembly failed for %s: %s", cell_id, exc,
                )

        # --- CV pipeline at zoom 16 (replaces deprecated LLM structure phase) ---
        # After dual-pass grid scoring completes on a zoom-16 cell, hand it to
        # the CV pipeline, which owns water mask, habitat masks, the four
        # detectors, and the orchestrator. All zoom-18 tile fetches and CV
        # work happen inside ``run_cell_full``; ``skip_existing=True`` means
        # re-runs of cell_analyzer don't redo CV work that's already on disk.
        if is_structure_phase_cell and state.registry is not None:
            try:
                # TODO: thread an explicit area_id through _RunState so multi-
                # area discovery doesn't have to assume rookery_bay_v2. For now
                # the cell_analyzer entry point doesn't carry area_id (only the
                # display name area_name), and the only live consumer is
                # rookery_bay_v2, so we hard-default here.
                area_id = "rookery_bay_v2"
                # CV pipeline manages its own API tracking inside water_mask
                # and the detectors; we deliberately don't bump state.api_calls
                # for CV work since that budget is for LLM/Claude calls.
                result = run_cell_full(area_id, cell_id, skip_existing=True)
                if result.succeeded or result.skipped:
                    from readwater.areas import Area  # local import to avoid top-level cycle risk
                    cv_anchors_path = Area(area_id).cell(cell_id).anchors_json
                    analysis.structures = []
                    analysis.structure_analysis = {
                        "cv_anchors_path": str(cv_anchors_path),
                    }
                else:
                    logger.warning(
                        "CV pipeline did not produce anchors for %s: status=%s error=%s",
                        cell_id, result.status, result.error,
                    )
                    analysis.structures = []
                    analysis.structure_analysis = {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("CV pipeline failed for cell %s: %s", cell_id, exc)

    first_image = next(iter(provider_images.values()), None)
    cell = Cell(
        id=cell_id,
        parent_id=parent_cell_id,
        center=center,
        size_miles=size_miles,
        depth=depth,
        zoom_level=zoom,
        bbox=bbox,
        analysis=analysis,
        image_path=first_image,
        provider_images=provider_images,
    )

    # --- Record metadata ---
    meta_entry: dict = {
        "cell_id": cell_id,
        "center": list(center),
        "bbox": {
            "north": bbox.north, "south": bbox.south,
            "east": bbox.east, "west": bbox.west,
        },
        "zoom": zoom,
        "depth": depth,
        "parent_id": parent_cell_id,
        "size_miles": size_miles,
        "provider_images": dict(provider_images),
        "providers": list(provider_images.keys()),
    }
    if analysis.structure_analysis.get("cv_anchors_path"):
        meta_entry["cv_anchors_path"] = analysis.structure_analysis["cv_anchors_path"]
    if z16_bundle_path:
        meta_entry["context_bundle_path"] = z16_bundle_path
    state.metadata.append(meta_entry)

    all_cells = [cell]

    # --- Recurse into sub-cells ---
    # CV-rubric scoring is BINARY: each sub_score is either 5.0 (KEEP) or
    # 0.0 (DROP). The legacy ambiguous-score path (score == 3.0 -> fetch
    # child + LLM confirm_fishing_water) is therefore dead with the CV
    # rubric in place; it's been removed alongside the LLM imports.
    # Structure-phase cells (zoom 16) already had their CV pipeline run
    # via ``run_cell_full`` above (water mask, habitat masks, detectors,
    # orchestrator); recursion from there is capped by TERMINAL_ZOOM so
    # we never descend to zoom 18 through the recursive path.
    next_zoom = zoom + 2
    if depth < max_depth and next_zoom <= TERMINAL_ZOOM:
        for sub_score in analysis.sub_scores:
            # In dry_run mode, honor threshold directly (placeholder scores = 5)
            if state.dry_run and sub_score.score < state.threshold:
                continue
            # In live mode, CV rubric drops produce score 0.0; keeps produce 5.0.
            if not state.dry_run and sub_score.score < state.threshold:
                continue

            child_id = _make_cell_id(cell_id, sub_score.row, sub_score.col)
            child_bbox = _sub_cell_bbox(bbox, sub_score.row, sub_score.col, SECTIONS)

            self_lineage = LineageRef(
                cell_id=cell_id,
                zoom=zoom,
                depth=depth,
                center=center,
                bbox=bbox,
                image_path=cell.image_path,
                position_in_parent=position_in_parent,
            )
            # Carry our CellContext (if any) forward to children.
            child_ancestor_contexts = (
                {**ancestor_contexts, cell_id: analysis.context}
                if analysis.context is not None
                else ancestor_contexts
            )
            child_cells = await _analyze_recursive(
                center=sub_score.center,
                zoom=next_zoom,
                depth=depth + 1,
                cell_id=child_id,
                parent_cell_id=cell_id,
                parent_context=analysis.overall_summary,
                max_depth=max_depth,
                state=state,
                bbox=child_bbox,
                ancestor_lineage=ancestor_lineage + [self_lineage],
                ancestor_contexts=child_ancestor_contexts,
                position_in_parent=(sub_score.row, sub_score.col),
            )
            if child_cells:
                cell.children_ids.append(child_id)
                all_cells.extend(child_cells)

    return all_cells
