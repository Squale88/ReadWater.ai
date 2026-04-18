"""Structure-phase orchestrator (Phase 1.5: grid-cell edition).

State machine:
  DISCOVER -> RANK_AND_DEFER -> per anchor:
    PLAN_CAPTURE -> FETCH -> RESOLVE_CONTINUATION (loop-capped)
    -> IDENTIFY (grid cells for anchor/members/influence)
    -> VALIDATE_CELLS -> EXTRACT
    -> IDENTIFY_SUBZONES (grid cells for subzones)
    -> VALIDATE_SUBZONE_CELLS -> EXTRACT_SUBZONES
    -> INTERPRET -> ASSEMBLE
  -> FINALIZE

Phase 1.5 uses GridCellExtractor for all features (polygon = bbox of cell
union). Phase 2 replaces the region/corridor/edge_band extractors with SAM
variants — same cell output from Claude, better polygon from the extractor.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PIL import Image

from readwater.api.providers.base import ImageProvider
from readwater.models.structure import (
    AnchorStructure,
    DeferredAnchor,
    FailedIdentification,
    FishableSubzone,
    InfluenceZone,
    InterpretedGeometry,
    LocalComplex,
    MemberFeature,
    ObservedGeometry,
    OverlapEntry,
    SegmentationIssue,
    StructurePhaseResult,
)
from readwater.pipeline.structure import prompts as llm
from readwater.pipeline.structure.extractors import (
    ClickBoxExtractor,
    build_gridcell_registry,
    get_extractor,
    is_subzone_type_allowed,
    mode_for,
)
from readwater.pipeline.structure.grid_overlay import (
    cells_to_bbox,
    cells_to_centroids,
)
from readwater.pipeline.structure.geo import clip_polygon_to_rect, pixel_to_latlon
from readwater.pipeline.structure.mosaic import (
    TILE_PX,
    Z16_CELL_PX,
    Mosaic,
    MosaicState,
    convex_hull,
    expand_plan,
    expand_polygon,
    polygon_iou,
    render_annotated,
    select_z18_centers,
)
from readwater.pipeline.structure.seed_validator import (
    Verdict,
    validate_cells,
)

logger = logging.getLogger(__name__)


# --- State enum (documentary; the agent is linear, not table-driven) ---


class StructurePhase(str, Enum):
    DISCOVER = "discover"
    RANK_AND_DEFER = "rank_and_defer"
    PLAN_CAPTURE = "plan_capture"
    FETCH = "fetch"
    RESOLVE_CONTINUATION = "resolve_continuation"
    IDENTIFY = "identify"
    VALIDATE_CELLS = "validate_cells"
    EXTRACT = "extract"
    IDENTIFY_SUBZONES = "identify_subzones"
    VALIDATE_SUBZONE_CELLS = "validate_subzone_cells"
    EXTRACT_SUBZONES = "extract_subzones"
    INTERPRET = "interpret"
    ASSEMBLE = "assemble"
    FINALIZE = "finalize"


@dataclass
class StructureBudget:
    calls_per_anchor: int = 15
    tiles_per_anchor: int = 25
    max_anchors_per_cell: int = 3
    continuation_loop_cap: int = 2

    calls_used: int = 0
    tiles_used: int = 0

    anchor_calls: int = 0
    anchor_tiles: int = 0

    def start_anchor(self) -> None:
        self.anchor_calls = 0
        self.anchor_tiles = 0

    def charge_call(self) -> None:
        self.anchor_calls += 1
        self.calls_used += 1

    def charge_tile(self) -> None:
        self.anchor_tiles += 1
        self.tiles_used += 1

    def anchor_exhausted(self) -> bool:
        return (
            self.anchor_calls >= self.calls_per_anchor
            or self.anchor_tiles >= self.tiles_per_anchor
        )


@dataclass
class StructurePaths:
    cell_root: Path
    structures_root: Path
    discovery_image_path: Path
    registry_path: Path
    grid_overlay_dir: Path

    @classmethod
    def for_cell(cls, base_output_dir: Path, cell_id: str) -> StructurePaths:
        cell_root = base_output_dir / "cells" / cell_id
        structures_root = cell_root / "structures"
        return cls(
            cell_root=cell_root,
            structures_root=structures_root,
            discovery_image_path=structures_root / "anchors_discovery.png",
            registry_path=structures_root / "registry.json",
            grid_overlay_dir=structures_root / "grid_overlays",
        )


# --- Generic helpers ---


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _rank_anchor(a: dict) -> float:
    scale_w = 1.0 if a.get("scale", "minor") == "major" else 0.5
    return (
        float(a.get("confidence", 0.0))
        * scale_w
        * float(a.get("expected_relevance", 0.0))
    )


# --- Cell-based geometry construction ---


def _build_observed_from_cells(
    cells: list[str],
    parsed_cells: list[tuple[int, int]],
    mode: str,
    mosaic: Mosaic,
    registry: dict,
    image_ref: str = "mosaic",
    negative_cells: list[str] | None = None,
    grid_rows: int = 0,
    grid_cols: int = 0,
) -> ObservedGeometry | None:
    """Run the right extractor for `mode`, produce ObservedGeometry + latlon."""
    extractor = get_extractor(mode, registry)
    out = extractor.extract(
        mosaic.image,
        positive_points=parsed_cells,   # (row, col) pairs
        negative_points=[],             # negative cells supported separately below
    )
    clipped = clip_polygon_to_rect(
        [(float(x), float(y)) for x, y in out.pixel_polygon],
        mosaic.width,
        mosaic.height,
    )
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)

    pos_centroids = cells_to_centroids(
        cells, grid_rows, grid_cols, (mosaic.width, mosaic.height),
    )
    neg_centroids = cells_to_centroids(
        negative_cells or [], grid_rows, grid_cols, (mosaic.width, mosaic.height),
    )

    return ObservedGeometry(
        pixel_polygon=clipped,
        latlon_polygon=latlon,
        image_ref=image_ref,
        extractor=out.extractor_name,
        extraction_mode=mode,
        seed_cells=list(cells),
        grid_rows=grid_rows or None,
        grid_cols=grid_cols or None,
        seed_positive_points=pos_centroids,
        seed_negative_points=neg_centroids,
        confidence=out.confidence,
    )


def _build_interpreted_from_cells(
    cells_field,
    anchor_polygon_px: list[tuple[int, int]],
    mosaic: Mosaic,
    grid_rows: int,
    grid_cols: int,
) -> InterpretedGeometry | None:
    """Turn an influence_zone cells field into an InterpretedGeometry.

    Accepts:
      - list of cell labels
      - the string "hull_of_anchor"
    """
    if isinstance(cells_field, str) and cells_field.strip() == "hull_of_anchor":
        expand = int(round(0.10 * min(mosaic.width, mosaic.height)))
        poly = expand_polygon(convex_hull(anchor_polygon_px), max(expand, 0))
        source = "convex_hull_of_anchor"
        rationale = "LLM escape-hatch: hull around anchor"
    elif isinstance(cells_field, list):
        bbox = cells_to_bbox(
            cells_field, grid_rows, grid_cols, (mosaic.width, mosaic.height),
        )
        if bbox is None:
            return None
        x, y, w, h = bbox
        poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        source = "llm_polygon"
        rationale = ""
    else:
        return None

    clipped = clip_polygon_to_rect(
        [(float(x), float(y)) for x, y in poly],
        mosaic.width,
        mosaic.height,
    )
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)
    return InterpretedGeometry(
        pixel_polygon=clipped,
        latlon_polygon=latlon,
        image_ref="mosaic",
        source=source,
        rationale=rationale,
    )


# --- Discovery preview image ---


def _draw_discovery_preview(
    z16_image_path: str,
    anchors: list[dict],
    grid_rows: int,
    grid_cols: int,
    out_path: Path,
) -> None:
    from PIL import ImageDraw, ImageFont

    img = Image.open(z16_image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20,
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for a in anchors:
        cells = a.get("cells") or []
        bbox = cells_to_bbox(cells, grid_rows, grid_cols, img.size)
        if bbox is None:
            continue
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=(255, 215, 0), width=3)
        label = f"{a.get('anchor_id', '?')} {a.get('structure_type', '')}"
        draw.text((x + 4, y + 4), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))


def _anchor_center_latlon_from_cells(
    cells: list[str],
    grid_rows: int,
    grid_cols: int,
    z16_center: tuple[float, float],
) -> tuple[float, float]:
    """Compute anchor's center lat/lon from its z16 cell list."""
    bbox = cells_to_bbox(cells, grid_rows, grid_cols, (Z16_CELL_PX, Z16_CELL_PX))
    if bbox is None:
        return z16_center
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    return pixel_to_latlon(
        cx, cy, Z16_CELL_PX, z16_center[0], z16_center[1], 16,
    )


def _bbox_z16_from_cells(
    cells: list[str],
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    bbox = cells_to_bbox(cells, grid_rows, grid_cols, (Z16_CELL_PX, Z16_CELL_PX))
    if bbox is None:
        return (0, 0, Z16_CELL_PX, Z16_CELL_PX)
    return bbox


# --- Identify + regenerate-once wrapper ---


async def _identify_with_retry(
    *,
    identify_fn,
    fn_kwargs: dict,
    get_features_to_validate,
    budget: StructureBudget,
) -> dict | None:
    """Run identify; validate each feature's cells; regenerate once on failure."""
    budget.charge_call()
    try:
        resp = await identify_fn(**fn_kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("identify call failed: %s", e)
        return None

    issues = _collect_cell_issues(resp, get_features_to_validate)
    if not issues:
        return resp

    feedback = "; ".join(f"{fid}: {reason}" for fid, reason in issues)
    logger.info("cell validation failed; regenerating once (%s)", feedback[:200])
    if budget.anchor_exhausted():
        return resp
    budget.charge_call()
    retry_kwargs = dict(fn_kwargs)
    retry_kwargs["feedback_note"] = feedback
    try:
        resp2 = await identify_fn(**retry_kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("identify regenerate failed: %s", e)
        return resp
    return resp2


def _collect_cell_issues(resp: dict, get_features_to_validate) -> list[tuple[str, str]]:
    grid_rows = int(resp.get("_grid_rows", 0))
    grid_cols = int(resp.get("_grid_cols", 0))
    if not grid_rows or not grid_cols:
        return []
    issues = []
    for feature_id, cells, mode in get_features_to_validate(resp):
        r = validate_cells(cells, grid_rows, grid_cols, extraction_mode=mode)
        if r.verdict is not Verdict.PASS:
            issues.append((feature_id, r.reason))
    return issues


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
    extractor_registry: dict | None = None,  # kept for Phase 2 injection
) -> StructurePhaseResult:
    """Run the full structure phase for one confirmed zoom-16 cell."""
    budget = budget or StructureBudget()
    paths = StructurePaths.for_cell(Path(base_output_dir), cell_id)
    paths.structures_root.mkdir(parents=True, exist_ok=True)
    paths.grid_overlay_dir.mkdir(parents=True, exist_ok=True)

    result = StructurePhaseResult(cell_id=cell_id)

    # --- DISCOVER ---
    logger.info("[structure:%s] DISCOVER", cell_id)
    budget.charge_call()
    try:
        discovery = await llm.discover_anchors(
            z15_image_path, z16_image_path, parent_context, cell_center,
            coverage_miles, grid_out_dir=paths.grid_overlay_dir,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[structure:%s] discovery failed: %s", cell_id, e)
        result.truncated = True
        _write_registry(paths.registry_path, cell_id, cell_center, [], result)
        result.registry_path = str(paths.registry_path)
        return result

    raw_discover = discovery.pop("raw_response", "")
    if raw_discover:
        (paths.structures_root / "discovery.md").write_text(
            raw_discover, encoding="utf-8",
        )
    grid_rows_z16 = int(discovery.pop("_grid_rows", 0))
    grid_cols_z16 = int(discovery.pop("_grid_cols", 0))
    discovery.pop("_gridded_image_path", None)

    anchors_raw: list[dict] = list(discovery.get("anchors", []))
    seen_ids: set[str] = set()
    for a in anchors_raw:
        aid = a.get("anchor_id") or f"a{uuid.uuid4().hex[:6]}"
        while aid in seen_ids:
            aid = f"a{uuid.uuid4().hex[:6]}"
        a["anchor_id"] = aid
        seen_ids.add(aid)

    _draw_discovery_preview(
        z16_image_path, anchors_raw, grid_rows_z16, grid_cols_z16,
        paths.discovery_image_path,
    )

    if not anchors_raw:
        logger.info("[structure:%s] no anchors discovered", cell_id)
        _write_registry(paths.registry_path, cell_id, cell_center, [], result)
        result.registry_path = str(paths.registry_path)
        return result

    # --- RANK_AND_DEFER ---
    logger.info("[structure:%s] RANK_AND_DEFER (%d anchors)", cell_id, len(anchors_raw))
    ranked = sorted(
        ((a, _rank_anchor(a)) for a in anchors_raw),
        key=lambda t: t[1],
        reverse=True,
    )
    top = ranked[: budget.max_anchors_per_cell]
    deferred_raw = ranked[budget.max_anchors_per_cell:]

    for a, rank in deferred_raw:
        bbox_tuple = _bbox_z16_from_cells(
            a.get("cells", []), grid_rows_z16, grid_cols_z16,
        )
        result.deferred.append(DeferredAnchor(
            anchor_id=a["anchor_id"],
            structure_type=a.get("structure_type", "other"),
            scale=a.get("scale", "minor"),
            confidence=float(a.get("confidence", 0.0)),
            expected_relevance=float(a.get("expected_relevance", 0.0)),
            approx_bbox_px_z16=bbox_tuple,
            rationale=a.get("rationale", ""),
            rank=rank,
        ))

    # --- Per-anchor ---
    mosaic_state = MosaicState()
    for a, _rank in top:
        try:
            await _process_anchor(
                anchor=a,
                cell_id=cell_id,
                cell_center=cell_center,
                grid_rows_z16=grid_rows_z16,
                grid_cols_z16=grid_cols_z16,
                provider=provider,
                paths=paths,
                budget=budget,
                mosaic_state=mosaic_state,
                result=result,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "[structure:%s] anchor %s failed: %s", cell_id, a.get("anchor_id"), e,
            )
            result.truncated_ids.append(a.get("anchor_id", "?"))
            result.truncated = True

    _audit_overlaps(result)

    _write_registry(
        registry_path=paths.registry_path,
        cell_id=cell_id,
        cell_center=cell_center,
        discovered=anchors_raw,
        result=result,
    )
    result.registry_path = str(paths.registry_path)
    result.api_calls_used = budget.calls_used
    result.tiles_fetched = budget.tiles_used
    return result


# --- Per-anchor pipeline ---


async def _process_anchor(
    anchor: dict,
    cell_id: str,
    cell_center: tuple[float, float],
    grid_rows_z16: int,
    grid_cols_z16: int,
    provider: ImageProvider,
    paths: StructurePaths,
    budget: StructureBudget,
    mosaic_state: MosaicState,
    result: StructurePhaseResult,
) -> None:
    anchor_id: str = anchor["anchor_id"]
    anchor_dir = paths.structures_root / anchor_id
    anchor_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = anchor_dir / "tiles"

    budget.start_anchor()

    # --- PLAN_CAPTURE (deterministic, bbox from z16 cells) ---
    logger.info("[structure:%s:%s] PLAN_CAPTURE", cell_id, anchor_id)
    bbox = _bbox_z16_from_cells(
        anchor.get("cells", []), grid_rows_z16, grid_cols_z16,
    )
    ce = anchor.get("continuation_edges") or {}
    plan = select_z18_centers(
        anchor_bbox_px_z16=bbox,
        z16_center=cell_center,
        continuation_edges=ce,
    )

    # --- FETCH ---
    logger.info(
        "[structure:%s:%s] FETCH %dx%d tiles", cell_id, anchor_id, plan.rows, plan.cols,
    )

    def _tile_fetched() -> None:
        budget.charge_tile()
        mosaic_state.on_fetch()

    mosaic = await Mosaic.build(
        plan, provider, tiles_dir,
        tile_cache=mosaic_state.tile_cache, on_fetch=_tile_fetched,
    )
    mosaic_path = anchor_dir / "mosaic.png"
    mosaic.save(mosaic_path)

    # --- RESOLVE_CONTINUATION ---
    anchor_center_latlon = _anchor_center_latlon_from_cells(
        anchor.get("cells", []), grid_rows_z16, grid_cols_z16, cell_center,
    )
    expansions = 0
    while expansions < budget.continuation_loop_cap and not budget.anchor_exhausted():
        logger.info(
            "[structure:%s:%s] RESOLVE_CONTINUATION iter=%d",
            cell_id, anchor_id, expansions,
        )
        budget.charge_call()
        try:
            resolve = await llm.resolve_continuation(
                str(mosaic_path), anchor,
                mosaic.width, mosaic.height, mosaic.rows, mosaic.cols,
                anchor_center_latlon,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[structure:%s:%s] resolve_continuation failed: %s",
                cell_id, anchor_id, e,
            )
            break
        raw = resolve.pop("raw_response", "")
        if raw:
            (anchor_dir / f"resolve_{expansions}.md").write_text(raw, encoding="utf-8")
        if resolve.get("structure_resolved"):
            break
        extends = resolve.get("extends") or {}
        if not any(extends.values()):
            break
        new_plan = expand_plan(plan, anchor_center_latlon, extends)
        if new_plan.rows == plan.rows and new_plan.cols == plan.cols:
            break
        plan = new_plan
        mosaic = await Mosaic.build(
            plan, provider, tiles_dir,
            tile_cache=mosaic_state.tile_cache, on_fetch=_tile_fetched,
        )
        mosaic.save(mosaic_path)
        expansions += 1

    if budget.anchor_exhausted():
        result.truncated_ids.append(anchor_id)
        result.truncated = True
        return

    # --- IDENTIFY + VALIDATE ---
    logger.info("[structure:%s:%s] IDENTIFY", cell_id, anchor_id)
    identify_resp = await _identify_with_retry(
        identify_fn=llm.identify_anchor,
        fn_kwargs={
            "mosaic_image_path": str(mosaic_path),
            "anchor": anchor,
            "grid_out_dir": paths.grid_overlay_dir / anchor_id,
        },
        get_features_to_validate=_anchor_response_validation_cells,
        budget=budget,
    )
    if identify_resp is None:
        result.truncated_ids.append(anchor_id)
        result.truncated = True
        return
    raw = identify_resp.pop("raw_response", "")
    if raw:
        (anchor_dir / "identify.md").write_text(raw, encoding="utf-8")
    grid_rows_m = int(identify_resp.pop("_grid_rows", 0))
    grid_cols_m = int(identify_resp.pop("_grid_cols", 0))
    identify_resp.pop("_gridded_image_path", None)

    registry = build_gridcell_registry(grid_rows_m, grid_cols_m, (mosaic.width, mosaic.height))

    # --- EXTRACT anchor ---
    anchor_block = identify_resp.get("anchor") or {}
    anchor_cells = anchor_block.get("cells") or []
    anchor_mode = mode_for(anchor.get("structure_type", "other"), "anchor")
    anchor_val = validate_cells(anchor_cells, grid_rows_m, grid_cols_m, anchor_mode)
    if anchor_val.verdict is not Verdict.PASS or not anchor_val.parsed_cells:
        _fail_identification(
            result, anchor_id, "anchor",
            f"anchor cells failed validation after retry: {anchor_val.reason}",
            regeneration_attempted=True,
        )
        return

    anchor_geom = _build_observed_from_cells(
        cells=anchor_cells,
        parsed_cells=anchor_val.parsed_cells,
        mode=anchor_mode,
        mosaic=mosaic,
        registry=registry,
        grid_rows=grid_rows_m,
        grid_cols=grid_cols_m,
    )
    if anchor_geom is None:
        anchor_geom = _fallback_region_geometry(
            anchor_val.parsed_cells, mosaic, result, anchor_id, "anchor",
            reason="anchor gridcell polygon collapsed",
            primary_extractor="gridcell",
            grid_rows=grid_rows_m,
            grid_cols=grid_cols_m,
            cells=anchor_cells,
        )
        if anchor_geom is None:
            _fail_identification(
                result, anchor_id, "anchor",
                "anchor polygon collapsed even after fallback",
                regeneration_attempted=True,
            )
            return

    # --- EXTRACT complex members ---
    complex_block = identify_resp.get("local_complex") or {}
    raw_members = complex_block.get("members") or []
    member_features: list[MemberFeature] = []
    for idx, m in enumerate(raw_members):
        m_id = f"{anchor_id}-m{idx + 1}"
        m_type = m.get("feature_type", "bar")
        m_cells = m.get("cells") or []
        m_mode = mode_for(m_type, "anchor")
        mval = validate_cells(m_cells, grid_rows_m, grid_cols_m, m_mode)
        if mval.verdict is not Verdict.PASS or not mval.parsed_cells:
            _fail_identification(
                result, m_id, "complex_member",
                f"member cells failed validation: {mval.reason}",
                regeneration_attempted=True,
            )
            continue
        m_geom = _build_observed_from_cells(
            cells=m_cells,
            parsed_cells=mval.parsed_cells,
            mode=m_mode,
            mosaic=mosaic,
            registry=registry,
            grid_rows=grid_rows_m,
            grid_cols=grid_cols_m,
        )
        if m_geom is None:
            m_geom = _fallback_region_geometry(
                mval.parsed_cells, mosaic, result, m_id, "complex_member",
                reason="member gridcell polygon collapsed",
                primary_extractor="gridcell",
                grid_rows=grid_rows_m,
                grid_cols=grid_cols_m,
                cells=m_cells,
            )
            if m_geom is None:
                _fail_identification(
                    result, m_id, "complex_member",
                    "member polygon collapsed even after fallback",
                    regeneration_attempted=True,
                )
                continue
        member_features.append(MemberFeature(
            name=m.get("name", f"member_{idx + 1}"),
            feature_type=m_type,
            geometry=m_geom,
            notes=m.get("notes", ""),
        ))

    # --- INTERPRET influence zone ---
    influence_block = identify_resp.get("influence_zone") or {}
    influence_cells_field = influence_block.get("cells")
    influence_geom = _build_interpreted_from_cells(
        influence_cells_field, anchor_geom.pixel_polygon, mosaic,
        grid_rows_m, grid_cols_m,
    )
    if influence_geom is None:
        # LLM influence polygon collapsed; fall back to the hull-of-anchor path.
        influence_geom = _build_interpreted_from_cells(
            "hull_of_anchor", anchor_geom.pixel_polygon, mosaic,
            grid_rows_m, grid_cols_m,
        )
    if influence_geom is None:
        # Last resort: a trivial box around the anchor polygon
        _fail_identification(
            result, anchor_id, "anchor",
            "influence polygon collapsed and hull fallback also failed",
            regeneration_attempted=False,
        )
        return

    # --- IDENTIFY_SUBZONES + VALIDATE + EXTRACT ---
    subzones_out: list[FishableSubzone] = []
    if not budget.anchor_exhausted():
        logger.info("[structure:%s:%s] IDENTIFY_SUBZONES", cell_id, anchor_id)
        subzone_resp = await _identify_with_retry(
            identify_fn=llm.identify_subzones,
            fn_kwargs={
                "mosaic_image_path": str(mosaic_path),
                "anchor": anchor,
                "grid_out_dir": paths.grid_overlay_dir / anchor_id,
            },
            get_features_to_validate=_subzone_response_validation_cells,
            budget=budget,
        )
        if subzone_resp is not None:
            raw_sz = subzone_resp.pop("raw_response", "")
            if raw_sz:
                (anchor_dir / "subzones.md").write_text(raw_sz, encoding="utf-8")
            grid_rows_s = int(subzone_resp.pop("_grid_rows", grid_rows_m))
            grid_cols_s = int(subzone_resp.pop("_grid_cols", grid_cols_m))
            subzone_resp.pop("_gridded_image_path", None)
            sz_registry = build_gridcell_registry(
                grid_rows_s, grid_cols_s, (mosaic.width, mosaic.height),
            )

            for sz in (subzone_resp.get("subzones") or [])[:4]:
                sz_id = sz.get("subzone_id") or f"{anchor_id}-s{len(subzones_out) + 1}"
                sz_type = sz.get("subzone_type", "other")
                if not is_subzone_type_allowed(sz_type):
                    _fail_identification(
                        result, sz_id, "subzone",
                        f"subzone_type '{sz_type}' not in v1 whitelist",
                        regeneration_attempted=False,
                    )
                    continue
                sz_cells = sz.get("cells") or []
                sz_mode = mode_for(sz_type, "subzone")
                szval = validate_cells(sz_cells, grid_rows_s, grid_cols_s, sz_mode)
                if szval.verdict is not Verdict.PASS or not szval.parsed_cells:
                    _fail_identification(
                        result, sz_id, "subzone",
                        f"subzone cells failed validation: {szval.reason}",
                        regeneration_attempted=True,
                    )
                    continue
                sz_geom = _build_observed_from_cells(
                    cells=sz_cells,
                    parsed_cells=szval.parsed_cells,
                    mode=sz_mode,
                    mosaic=mosaic,
                    registry=sz_registry,
                    grid_rows=grid_rows_s,
                    grid_cols=grid_cols_s,
                )
                if sz_geom is None:
                    sz_geom = _fallback_region_geometry(
                        szval.parsed_cells, mosaic, result, sz_id, "subzone",
                        reason="subzone gridcell polygon collapsed",
                        primary_extractor="gridcell",
                        grid_rows=grid_rows_s,
                        grid_cols=grid_cols_s,
                        cells=sz_cells,
                    )
                    if sz_geom is None:
                        continue
                subzones_out.append(FishableSubzone(
                    subzone_id=sz_id,
                    anchor_id=anchor_id,
                    subzone_type=sz_type,
                    geometry=sz_geom,
                    relative_priority=_clamp(
                        float(sz.get("relative_priority", 0.5)), 0.0, 1.0,
                    ),
                    reasoning_summary=sz.get("reasoning_summary", ""),
                    confidence=_clamp(float(sz.get("confidence", 0.5)), 0.0, 1.0),
                ))

    # --- ASSEMBLE ---
    anchor_obj = AnchorStructure(
        anchor_id=anchor_id,
        structure_type=anchor.get("structure_type", "other"),
        scale=anchor.get("scale", "minor"),
        anchor_center_latlon=anchor_center_latlon,
        geometry=anchor_geom,
        orientation_deg=None,
        confidence=_clamp(float(anchor.get("confidence", 0.0)), 0.0, 1.0),
        rationale=anchor.get("rationale", ""),
        source_images_used=[str(mosaic_path)],
    )
    complex_obj = LocalComplex(
        complex_id=f"{anchor_id}-complex",
        anchor_id=anchor_id,
        members=member_features,
        relationship_summary=complex_block.get("relationship_summary", ""),
        envelope=None,
    )
    influence_obj = InfluenceZone(
        influence_zone_id=f"{anchor_id}-influence",
        anchor_id=anchor_id,
        geometry=influence_geom,
        influence_shape_type=influence_block.get("shape_type", "radial"),
        bounded_by=list(influence_block.get("bounded_by", [])),
        dominance_strength=_clamp(
            float(influence_block.get("dominance_strength", 0.5)), 0.0, 1.0,
        ),
        competing_structures=list(influence_block.get("competing_structures", [])),
    )

    result.anchors.append(anchor_obj)
    result.complexes.append(complex_obj)
    result.influences.append(influence_obj)
    result.subzones.extend(subzones_out)

    # --- Render annotated PNG + per-anchor JSON ---
    anchor_polys = [(anchor_obj.anchor_id, list(anchor_geom.pixel_polygon))]
    complex_polys = [
        (f"{anchor_id}:{m.name}", list(m.geometry.pixel_polygon))
        for m in member_features
    ]
    influence_polys = [(f"{anchor_id}:influence", list(influence_geom.pixel_polygon))]
    subzone_polys = [
        (sz.subzone_id, list(sz.geometry.pixel_polygon)) for sz in subzones_out
    ]
    all_seed_points = _collect_all_seed_points(
        anchor_geom, member_features, subzones_out,
    )
    annotated_path = anchor_dir / "annotated.png"
    render_annotated(
        base_image=mosaic.image,
        out_path=annotated_path,
        anchor_polygons_px=anchor_polys,
        complex_polygons_px=complex_polys,
        influence_polygons_px=influence_polys,
        subzone_polygons_px=subzone_polys,
        seed_points_px=all_seed_points,
    )
    result.annotated_image_paths[anchor_id] = str(annotated_path)
    result.mosaic_image_paths[anchor_id] = str(mosaic_path)

    per_anchor = {
        "anchor": anchor_obj.model_dump(),
        "local_complex": complex_obj.model_dump(),
        "influence_zone": influence_obj.model_dump(),
        "subzones": [sz.model_dump() for sz in subzones_out],
    }
    (anchor_dir / "result.json").write_text(
        json.dumps(per_anchor, indent=2), encoding="utf-8",
    )


# --- Response → (feature_id, cells, mode) iterators for validation ---


def _anchor_response_validation_cells(resp: dict):
    anchor = resp.get("anchor") or {}
    yield ("anchor", anchor.get("cells", []), "region")

    for idx, m in enumerate((resp.get("local_complex") or {}).get("members") or []):
        m_type = m.get("feature_type", "bar")
        m_mode = mode_for(m_type, "anchor")
        yield (f"member_{idx + 1}", m.get("cells", []), m_mode)


def _subzone_response_validation_cells(resp: dict):
    for sz in resp.get("subzones") or []:
        sz_type = sz.get("subzone_type", "other")
        mode = mode_for(sz_type, "subzone")
        yield (sz.get("subzone_id", sz_type), sz.get("cells", []), mode)


# --- Fallbacks + logging ---


def _fallback_region_geometry(
    parsed_cells: list[tuple[int, int]],
    mosaic: Mosaic,
    result: StructurePhaseResult,
    feature_id: str,
    feature_level: str,
    reason: str,
    primary_extractor: str,
    grid_rows: int,
    grid_cols: int,
    cells: list[str],
) -> ObservedGeometry | None:
    """Fall back to ClickBoxExtractor(region) seeded by cell centroids."""
    centroids = cells_to_centroids(
        cells, grid_rows, grid_cols, (mosaic.width, mosaic.height),
    )
    if not centroids:
        return None
    fb = ClickBoxExtractor(mode="region")
    out = fb.extract(mosaic.image, centroids, [])
    clipped = clip_polygon_to_rect(
        [(float(x), float(y)) for x, y in out.pixel_polygon],
        mosaic.width, mosaic.height,
    )
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)
    result.segmentation_issues.append(SegmentationIssue(
        feature_id=feature_id,
        feature_level=feature_level,
        extractor_attempted=primary_extractor,
        fallback_used="clickbox_region",
        reason=reason,
    ))
    return ObservedGeometry(
        pixel_polygon=clipped,
        latlon_polygon=latlon,
        image_ref="mosaic",
        extractor="fallback",
        extraction_mode="region",
        seed_cells=list(cells),
        grid_rows=grid_rows or None,
        grid_cols=grid_cols or None,
        seed_positive_points=centroids,
        seed_negative_points=[],
        confidence=None,
    )


def _fail_identification(
    result: StructurePhaseResult,
    feature_id: str,
    feature_level: str,
    reason: str,
    regeneration_attempted: bool,
) -> None:
    result.failed_identifications.append(FailedIdentification(
        feature_id=feature_id,
        feature_level=feature_level,
        reason=reason,
        regeneration_attempted=regeneration_attempted,
    ))


def _collect_all_seed_points(
    anchor_geom: ObservedGeometry,
    members: list[MemberFeature],
    subzones: list[FishableSubzone],
) -> list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]]:
    out: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]] = [
        ("anchor", list(anchor_geom.seed_positive_points), list(anchor_geom.seed_negative_points)),
    ]
    for m in members:
        out.append((
            m.name,
            list(m.geometry.seed_positive_points),
            list(m.geometry.seed_negative_points),
        ))
    for sz in subzones:
        out.append((
            sz.subzone_id,
            list(sz.geometry.seed_positive_points),
            list(sz.geometry.seed_negative_points),
        ))
    return out


# --- Overlap audit ---


def _audit_overlaps(result: StructurePhaseResult) -> None:
    if not result.anchors:
        return
    canvas = (TILE_PX * 5, TILE_PX * 5)

    subordinated: set[str] = set()
    for i, a in enumerate(result.anchors):
        for b in result.anchors[i + 1:]:
            if a.anchor_id in subordinated or b.anchor_id in subordinated:
                continue
            iou = polygon_iou(
                list(a.geometry.pixel_polygon),
                list(b.geometry.pixel_polygon),
                canvas,
            )
            if iou <= 0.1:
                continue
            if a.confidence >= b.confidence:
                keep, drop = a, b
            else:
                keep, drop = b, a
            policy = "subordinated" if iou > 0.25 else "kept"
            if iou > 0.25:
                subordinated.add(drop.anchor_id)
            result.overlap_report.append(OverlapEntry(
                anchor_id_a=keep.anchor_id, anchor_id_b=drop.anchor_id,
                level="anchor", iou=iou, policy=policy,
            ))

    if subordinated:
        result.influences = [
            inf for inf in result.influences if inf.anchor_id not in subordinated
        ]
        result.subordinated_ids.extend(subordinated)

    _log_iou_overlaps(result, "influence", result.influences, canvas)
    _log_iou_overlaps(result, "subzone", result.subzones, canvas)


def _log_iou_overlaps(
    result: StructurePhaseResult,
    level: str,
    items: list,
    canvas: tuple[int, int],
) -> None:
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            iou = polygon_iou(
                list(a.geometry.pixel_polygon),
                list(b.geometry.pixel_polygon),
                canvas,
            )
            if iou <= 0.1:
                continue
            id_a = (
                getattr(a, "anchor_id", None)
                or getattr(a, "influence_zone_id", None)
                or getattr(a, "subzone_id", "?")
            )
            id_b = (
                getattr(b, "anchor_id", None)
                or getattr(b, "influence_zone_id", None)
                or getattr(b, "subzone_id", "?")
            )
            result.overlap_report.append(OverlapEntry(
                anchor_id_a=str(id_a), anchor_id_b=str(id_b),
                level=level, iou=iou, policy="kept",
            ))


# --- Registry ---


def _write_registry(
    registry_path: Path,
    cell_id: str,
    cell_center: tuple[float, float],
    discovered: list[dict],
    result: StructurePhaseResult,
) -> None:
    payload = {
        "cell_id": cell_id,
        "cell_center_latlon": list(cell_center),
        "discovered": discovered,
        "processed": [a.anchor_id for a in result.anchors],
        "deferred": [d.model_dump() for d in result.deferred],
        "truncated": list(result.truncated_ids),
        "subordinated": list(result.subordinated_ids),
        "overlap_report": [o.model_dump() for o in result.overlap_report],
        "segmentation_issues": [s.model_dump() for s in result.segmentation_issues],
        "failed_identifications": [
            f.model_dump() for f in result.failed_identifications
        ],
        "final_accepted": {
            "anchors": [
                a.anchor_id for a in result.anchors
                if a.anchor_id not in result.subordinated_ids
            ],
            "complexes": [c.complex_id for c in result.complexes],
            "influences": [i.influence_zone_id for i in result.influences],
            "subzones": [s.subzone_id for s in result.subzones],
        },
        "api_calls_used": result.api_calls_used,
        "tiles_fetched": result.tiles_fetched,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "StructurePhase",
    "StructureBudget",
    "StructurePaths",
    "run_structure_phase",
]
