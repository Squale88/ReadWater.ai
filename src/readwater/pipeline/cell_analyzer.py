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

from readwater.api.claude_vision import (
    MODEL,
    _cell_number_to_row_col,
    analyze_grid_image,
    confirm_fishing_water,
)
from readwater.api.providers.registry import ImageProviderRegistry
from readwater.models.cell import BoundingBox, Cell, CellAnalysis, CellScore
from readwater.pipeline.image_processing import draw_grid_overlay
from readwater.pipeline.structure import run_structure_phase
from readwater.pipeline.structure.agent import StructureBudget

logger = logging.getLogger(__name__)

# --- Constants ---

MILES_PER_DEG_LAT = 69.0
EARTH_CIRCUMFERENCE_MILES = 24901.0
SECTIONS = 4
TERMINAL_ZOOM = 16  # structure phase owns zoom 18 and below
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
    structure_budget: StructureBudget = field(default_factory=StructureBudget)


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
    structure_budget: StructureBudget | None = None,
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
        structure_budget=structure_budget or StructureBudget(),
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
) -> list[Cell]:
    """Internal recursive implementation of cell analysis."""
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

        # --- Confirmation gate (depth > 0 only) ---
        # Parent's grid scoring flagged this cell. Before spending an API call
        # on grid analysis, confirm there's actually fishing water here.
        if depth > 0:
            confirm_result = await confirm_fishing_water(
                first_image, parent_context, center, size_miles,
            )
            raw_confirm = confirm_result.pop("raw_response", "")
            if raw_confirm:
                md_path = Path(first_image).with_name(
                    Path(first_image).stem + "_confirm.md"
                )
                md_path.write_text(raw_confirm, encoding="utf-8")

            if not confirm_result.get("has_fishing_water", False):
                logger.info(
                    "Confirmation rejected cell %s: %s",
                    cell_id, confirm_result.get("reasoning", ""),
                )
                analysis = CellAnalysis(
                    overall_summary=f"Rejected: {confirm_result.get('reasoning', 'no fishing water')}",
                    sub_scores=[
                        CellScore(row=r, col=c, score=0.0, summary="Rejected by confirmation", center=ctr)
                        for r, c, ctr in sub_cells_geo
                    ],
                    model_used=MODEL,
                )
                # Build cell with rejection, skip grid scoring and recursion
                first_img_path = next(iter(provider_images.values()), None)
                cell = Cell(
                    id=cell_id, parent_id=parent_cell_id, center=center,
                    size_miles=size_miles, depth=depth, zoom_level=zoom,
                    bbox=bbox, analysis=analysis, image_path=first_img_path,
                    provider_images=provider_images,
                )
                state.metadata.append({
                    "cell_id": cell_id, "center": list(center),
                    "bbox": {"north": bbox.north, "south": bbox.south, "east": bbox.east, "west": bbox.west},
                    "zoom": zoom, "depth": depth, "parent_id": parent_cell_id,
                    "size_miles": size_miles, "provider_images": dict(provider_images),
                    "providers": list(provider_images.keys()), "confirmed": False,
                })
                return [cell]

        # --- Grid scoring (all non-rejected cells) ---
        grid_path = draw_grid_overlay(first_image)
        try:
            vision_result = await analyze_grid_image(
                grid_path, parent_context, zoom, center, size_miles,
            )
            raw_response = vision_result.pop("raw_response", "")
            if raw_response:
                md_path = Path(grid_path).with_suffix(".md")
                md_path.write_text(raw_response, encoding="utf-8")
            scored_cells = []
            for sc in vision_result.get("sub_scores", []):
                row, col = _cell_number_to_row_col(sc["cell_number"])
                scored_cells.append(CellScore(
                    row=row,
                    col=col,
                    score=float(sc["score"]),
                    summary=sc.get("reasoning", ""),
                    center=sub_cells_map.get((row, col), center),
                ))
            analysis = CellAnalysis(
                overall_summary=vision_result.get("summary", ""),
                sub_scores=scored_cells,
                hydrology_notes=vision_result.get("hydrology_notes", ""),
                model_used=MODEL,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Grid analysis failed for cell %s: %s", cell_id, e)
            analysis = CellAnalysis(
                overall_summary=f"Analysis failed: {e}",
                sub_scores=[
                    CellScore(row=r, col=c, score=0.0, summary="Analysis error", center=ctr)
                    for r, c, ctr in sub_cells_geo
                ],
                model_used=MODEL,
            )

        # --- Structure phase at zoom 16 (replaces old zoom-18 terminal) ---
        if is_structure_phase_cell and state.registry is not None:
            try:
                z15_provider = state.registry.get_default_provider(Z15_CONTEXT_PROVIDER_ROLE)
                z15_path = str(
                    state.output_dir / _image_filename(cell_id, depth, "z15ctx")
                )
                if state.api_calls >= state.max_api_calls:
                    logger.warning(
                        "API limit reached before zoom-15 context fetch for %s", cell_id,
                    )
                else:
                    await asyncio.sleep(0.5)
                    await z15_provider.fetch(center, 15, z15_path)
                    state.api_calls += 1

                    structure_provider = state.registry.get_providers("structure")[0]
                    structure_result = await run_structure_phase(
                        cell_id=cell_id,
                        cell_center=center,
                        z15_image_path=z15_path,
                        z16_image_path=first_image,
                        provider=structure_provider,
                        base_output_dir=state.output_dir,
                        parent_context=parent_context,
                        coverage_miles=size_miles,
                        budget=state.structure_budget,
                    )
                    state.api_calls += structure_result.api_calls_used
                    analysis.structures = list(structure_result.anchors)
                    analysis.structure_analysis = {
                        "anchors": [a.model_dump() for a in structure_result.anchors],
                        "complexes": [c.model_dump() for c in structure_result.complexes],
                        "influences": [i.model_dump() for i in structure_result.influences],
                        "subzones": [s.model_dump() for s in structure_result.subzones],
                        "deferred": [d.model_dump() for d in structure_result.deferred],
                        "registry_path": structure_result.registry_path,
                        "truncated": structure_result.truncated,
                    }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Structure phase failed for cell %s: %s", cell_id, exc)

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
    if analysis.structures:
        meta_entry["structures_summary"] = [
            {
                "anchor_id": a.anchor_id,
                "structure_type": a.structure_type,
                "scale": a.scale,
                "confidence": a.confidence,
            }
            for a in analysis.structures
        ]
    state.metadata.append(meta_entry)

    all_cells = [cell]

    # --- Recurse into sub-cells above threshold ---
    # Structure-phase cells (zoom 16) own their own deeper investigation; do
    # not descend to zoom 18 through the recursive path.
    next_zoom = zoom + 2
    if depth < max_depth and next_zoom <= TERMINAL_ZOOM:
        for sub_score in analysis.sub_scores:
            if sub_score.score >= state.threshold:
                child_id = _make_cell_id(cell_id, sub_score.row, sub_score.col)
                child_bbox = _sub_cell_bbox(bbox, sub_score.row, sub_score.col, SECTIONS)
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
                )
                if child_cells:
                    cell.children_ids.append(child_id)
                    all_cells.extend(child_cells)

    return all_cells
