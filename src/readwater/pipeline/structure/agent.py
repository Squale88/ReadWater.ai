"""Structure-phase orchestrator — the investigator state machine.

Replaces the old zoom-18 prose terminal call. Starting from a confirmed
zoom-16 cell plus its zoom-15 parent-context image, this agent:

  DISCOVER -> RANK_AND_DEFER -> PLAN_CAPTURE (deterministic) -> FETCH ->
  RESOLVE_CONTINUATION [loops back to FETCH] ->
  MODEL_INFLUENCE -> DEFINE_SUBZONES -> FINALIZE

Produces durable zone objects (AnchorStructure, LocalComplex, InfluenceZone,
FishableSubzone) with both pixel and lat/lon polygons, plus an annotated PNG
per anchor and a single cell-level registry.json.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from readwater.api.providers.base import ImageProvider
from readwater.models.structure import (
    AnchorStructure,
    DeferredAnchor,
    FishableSubzone,
    InfluenceZone,
    LocalComplex,
    OverlapEntry,
    StructurePhaseResult,
    ZoneGeometry,
)
from readwater.pipeline.structure import prompts as llm
from readwater.pipeline.structure.geo import clip_polygon_to_rect
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

logger = logging.getLogger(__name__)


class StructurePhase(str, Enum):
    DISCOVER = "discover"
    RANK_AND_DEFER = "rank_and_defer"
    PLAN_CAPTURE = "plan_capture"
    FETCH = "fetch"
    RESOLVE_CONTINUATION = "resolve_continuation"
    MODEL_INFLUENCE = "model_influence"
    DEFINE_SUBZONES = "define_subzones"
    FINALIZE = "finalize"


@dataclass
class StructureBudget:
    """Per-run budget for the structure phase."""

    calls_per_anchor: int = 15
    tiles_per_anchor: int = 25
    max_anchors_per_cell: int = 3
    continuation_loop_cap: int = 2

    calls_used: int = 0
    tiles_used: int = 0

    # Per-anchor rolling counters (reset at the start of each anchor)
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
    """Where structure-phase artifacts get written for one cell."""

    cell_root: Path
    structures_root: Path
    discovery_image_path: Path
    registry_path: Path

    @classmethod
    def for_cell(cls, base_output_dir: Path, cell_id: str) -> StructurePaths:
        cell_root = base_output_dir / "cells" / cell_id
        structures_root = cell_root / "structures"
        return cls(
            cell_root=cell_root,
            structures_root=structures_root,
            discovery_image_path=structures_root / "anchors_discovery.png",
            registry_path=structures_root / "registry.json",
        )


# --- Helpers ---


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _rank_anchor(a: dict) -> float:
    scale_w = 1.0 if a.get("scale", "minor") == "major" else 0.5
    return float(a.get("confidence", 0.0)) * scale_w * float(a.get("expected_relevance", 0.0))


def _frac_polygon_to_px(
    polygon: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Scale a normalized [0-1] polygon to pixel coords for a given image."""
    return [(float(p[0]) * width, float(p[1]) * height) for p in polygon]


def _clip_and_convert(
    polygon_px: list[list[float]] | list[tuple[float, float]],
    mosaic: Mosaic,
    image_ref: str,
) -> ZoneGeometry | None:
    """Clip a pixel polygon to mosaic bounds and lift it to a ZoneGeometry. None if collapsed."""
    pts = [(float(p[0]), float(p[1])) for p in polygon_px]
    clipped = clip_polygon_to_rect(pts, mosaic.width, mosaic.height)
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)
    return ZoneGeometry(pixel_polygon=clipped, latlon_polygon=latlon, image_ref=image_ref)


def _resolve_influence_polygon(
    influence_field: Any,
    anchor_polygon_px: list[tuple[int, int]],
    mosaic_width: int,
    mosaic_height: int,
) -> list[tuple[int, int]]:
    """Normalize the influence_zone polygon field to pixel coords.

    Accepts:
      - list of [x, y] normalized fractions (preferred, new schema)
      - escape-hatch dict {"convex_hull_of_anchor": true, "expand_frac"/"expand_px": num}
      - legacy list of [x, y] pixel values (values > 1 trigger pixel interpretation)
    """
    if isinstance(influence_field, dict) and influence_field.get("convex_hull_of_anchor"):
        hull = convex_hull(anchor_polygon_px)
        if "expand_frac" in influence_field:
            expand = int(round(float(influence_field["expand_frac"]) * min(mosaic_width, mosaic_height)))
        else:
            expand = int(influence_field.get("expand_px", 100))
        return expand_polygon(hull, max(expand, 0))
    if isinstance(influence_field, list):
        return _normalize_or_pixel_polygon(influence_field, mosaic_width, mosaic_height)
    return []


def _normalize_or_pixel_polygon(
    polygon: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Interpret a polygon as normalized [0-1] or pixel coords.

    If every coordinate is <= 1.0, treat as normalized and scale to pixel space.
    Otherwise treat as already-pixel. This keeps legacy pixel responses working
    while new responses use normalized coords.
    """
    if not polygon:
        return []
    flat = [v for pt in polygon for v in pt]
    if flat and max(flat) <= 1.0:
        scaled = _frac_polygon_to_px(polygon, width, height)
        return [(int(round(x)), int(round(y))) for (x, y) in scaled]
    return [(int(round(p[0])), int(round(p[1]))) for p in polygon]


def _draw_discovery_preview(
    z16_image_path: str,
    anchors: list[dict],
    out_path: Path,
) -> None:
    """Outline discovered anchors on the zoom-16 cell image for debugging."""
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
        frac = a.get("approx_bbox_frac")
        if frac and max(float(v) for v in frac) <= 1.5:
            x, y, w, h = [float(v) * Z16_CELL_PX for v in frac]
        else:
            bbox = a.get("approx_bbox_px_z16") or [0, 0, 0, 0]
            x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        draw.rectangle([x, y, x + w, y + h], outline=(255, 215, 0), width=3)
        label = f"{a.get('anchor_id', '?')} {a.get('structure_type', '')}"
        draw.text((x + 4, y + 4), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))


# --- Entry point ---


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
) -> StructurePhaseResult:
    """Run the full structure phase for one confirmed zoom-16 cell.

    Returns a populated StructurePhaseResult. Writes artifacts under
    `{base_output_dir}/cells/{cell_id}/structures/...`.
    """
    budget = budget or StructureBudget()
    paths = StructurePaths.for_cell(Path(base_output_dir), cell_id)
    paths.structures_root.mkdir(parents=True, exist_ok=True)

    result = StructurePhaseResult(cell_id=cell_id)

    # --- DISCOVER ---
    logger.info("[structure:%s] DISCOVER", cell_id)
    budget.charge_call()
    try:
        discovery = await llm.discover_anchors(
            z15_image_path, z16_image_path, parent_context, cell_center, coverage_miles,
        )
    except Exception as e:
        logger.warning("[structure:%s] discovery failed: %s", cell_id, e)
        result.truncated = True
        _write_registry(paths.registry_path, cell_id, cell_center, [], [], [], [], [], [], result)
        result.registry_path = str(paths.registry_path)
        return result

    raw_discover = discovery.pop("raw_response", "")
    if raw_discover:
        (paths.structures_root / "discovery.md").write_text(raw_discover, encoding="utf-8")

    anchors_raw: list[dict] = list(discovery.get("anchors", []))
    # Assign stable IDs if the LLM didn't.
    seen_ids: set[str] = set()
    for a in anchors_raw:
        aid = a.get("anchor_id") or f"a{uuid.uuid4().hex[:6]}"
        while aid in seen_ids:
            aid = f"a{uuid.uuid4().hex[:6]}"
        a["anchor_id"] = aid
        seen_ids.add(aid)

    _draw_discovery_preview(z16_image_path, anchors_raw, paths.discovery_image_path)

    if not anchors_raw:
        logger.info("[structure:%s] no anchors discovered", cell_id)
        _write_registry(paths.registry_path, cell_id, cell_center, [], [], [], [], [], [], result)
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
        # Accept normalized bbox OR legacy pixel bbox; normalize to pixel for storage.
        frac = a.get("approx_bbox_frac")
        if frac and max(float(v) for v in frac) <= 1.5:
            bbox_tuple = (
                int(round(float(frac[0]) * Z16_CELL_PX)),
                int(round(float(frac[1]) * Z16_CELL_PX)),
                int(round(float(frac[2]) * Z16_CELL_PX)),
                int(round(float(frac[3]) * Z16_CELL_PX)),
            )
        else:
            raw = a.get("approx_bbox_px_z16") or [0, 0, 0, 0]
            bbox_tuple = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
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

    # --- Loop per processed anchor ---
    mosaic_state = MosaicState()
    for a, rank in top:
        try:
            await _process_anchor(
                anchor=a,
                rank=rank,
                cell_id=cell_id,
                cell_center=cell_center,
                provider=provider,
                paths=paths,
                budget=budget,
                mosaic_state=mosaic_state,
                result=result,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[structure:%s] anchor %s failed: %s", cell_id, a.get("anchor_id"), e,
            )
            result.truncated_ids.append(a.get("anchor_id", "?"))
            result.truncated = True

    # --- Overlap audit + anchor subordination ---
    _audit_overlaps(result)

    # --- Registry ---
    _write_registry(
        registry_path=paths.registry_path,
        cell_id=cell_id,
        cell_center=cell_center,
        discovered=anchors_raw,
        processed=[a.anchor_id for a in result.anchors],
        deferred_entries=result.deferred,
        truncated_ids=result.truncated_ids,
        failed_geometry_ids=result.failed_geometry_ids,
        overlap_report=result.overlap_report,
        result=result,
    )
    result.registry_path = str(paths.registry_path)
    result.api_calls_used = budget.calls_used
    result.tiles_fetched = budget.tiles_used
    return result


# --- Per-anchor pipeline ---


async def _process_anchor(
    anchor: dict,
    rank: float,
    cell_id: str,
    cell_center: tuple[float, float],
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

    # --- PLAN_CAPTURE (deterministic) ---
    logger.info("[structure:%s:%s] PLAN_CAPTURE", cell_id, anchor_id)
    # Prefer normalized bbox; fall back to pixel bbox for legacy responses.
    bbox_frac = anchor.get("approx_bbox_frac")
    if bbox_frac and max(float(v) for v in bbox_frac) <= 1.5:
        bbox = (
            int(round(float(bbox_frac[0]) * Z16_CELL_PX)),
            int(round(float(bbox_frac[1]) * Z16_CELL_PX)),
            int(round(float(bbox_frac[2]) * Z16_CELL_PX)),
            int(round(float(bbox_frac[3]) * Z16_CELL_PX)),
        )
    else:
        raw = anchor.get("approx_bbox_px_z16") or [0, 0, Z16_CELL_PX, Z16_CELL_PX]
        bbox = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
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
    if plan.rows * plan.cols + budget.anchor_tiles > budget.tiles_per_anchor:
        logger.warning(
            "[structure:%s:%s] plan exceeds per-anchor tile budget; truncating",
            cell_id, anchor_id,
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

    # --- RESOLVE_CONTINUATION (loop-capped) ---
    anchor_center_latlon = _anchor_center_latlon(bbox, cell_center)
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
            # Already at cap; nothing to do.
            break
        plan = new_plan
        mosaic = await Mosaic.build(
            plan, provider, tiles_dir,
            tile_cache=mosaic_state.tile_cache, on_fetch=_tile_fetched,
        )
        mosaic.save(mosaic_path)
        expansions += 1

    # --- MODEL_INFLUENCE ---
    if budget.anchor_exhausted():
        result.truncated_ids.append(anchor_id)
        result.truncated = True
        return

    logger.info("[structure:%s:%s] MODEL_INFLUENCE", cell_id, anchor_id)
    budget.charge_call()
    try:
        infl = await llm.model_influence(
            str(mosaic_path), anchor, mosaic.width, mosaic.height,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[structure:%s:%s] model_influence failed: %s", cell_id, anchor_id, e)
        result.truncated_ids.append(anchor_id)
        return
    raw = infl.pop("raw_response", "")
    if raw:
        (anchor_dir / "influence.md").write_text(raw, encoding="utf-8")

    # Accept new normalized key ("anchor_polygon") or legacy pixel key ("anchor_polygon_px").
    raw_anchor = infl.get("anchor_polygon") or infl.get("anchor_polygon_px") or []
    anchor_poly_px = _normalize_or_pixel_polygon(raw_anchor, mosaic.width, mosaic.height)
    anchor_geom = _clip_and_convert(anchor_poly_px, mosaic, image_ref="mosaic")
    if anchor_geom is None:
        logger.warning("[structure:%s:%s] anchor polygon collapsed", cell_id, anchor_id)
        result.failed_geometry_ids.append(anchor_id)
        return

    complex_block = infl.get("local_complex") or {}
    raw_complex = complex_block.get("polygon") or complex_block.get("polygon_px") or []
    complex_poly_px = _normalize_or_pixel_polygon(raw_complex, mosaic.width, mosaic.height)
    if len(complex_poly_px) < 3:
        # Fallback: buffer anchor convex hull by a small amount
        buf = int(round(0.05 * min(mosaic.width, mosaic.height)))
        complex_poly_px = expand_polygon(convex_hull(anchor_geom.pixel_polygon), buf)
    complex_geom = _clip_and_convert(complex_poly_px, mosaic, image_ref="mosaic")
    if complex_geom is None:
        complex_geom = anchor_geom  # degenerate: fall back to anchor

    influence_block = infl.get("influence_zone") or {}
    raw_poly = influence_block.get("polygon") or influence_block.get("polygon_px") or []
    influence_poly_px = _resolve_influence_polygon(
        raw_poly, anchor_geom.pixel_polygon, mosaic.width, mosaic.height,
    )
    influence_geom = _clip_and_convert(influence_poly_px, mosaic, image_ref="mosaic")
    if influence_geom is None:
        expand = int(round(0.10 * min(mosaic.width, mosaic.height)))
        influence_poly_px = expand_polygon(convex_hull(anchor_geom.pixel_polygon), expand)
        influence_geom = _clip_and_convert(influence_poly_px, mosaic, image_ref="mosaic")
    # At this point influence_geom should be non-None, but guard anyway.
    if influence_geom is None:
        result.failed_geometry_ids.append(anchor_id)
        return

    # --- DEFINE_SUBZONES ---
    if budget.anchor_exhausted():
        result.truncated_ids.append(anchor_id)
        result.truncated = True
        return

    logger.info("[structure:%s:%s] DEFINE_SUBZONES", cell_id, anchor_id)
    budget.charge_call()
    subzone_geoms: list[tuple[FishableSubzone, list[tuple[int, int]]]] = []
    try:
        sub_resp = await llm.define_subzones(
            str(mosaic_path), anchor, mosaic.width, mosaic.height,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[structure:%s:%s] define_subzones failed: %s", cell_id, anchor_id, e)
        sub_resp = {"subzones": []}
    raw = sub_resp.pop("raw_response", "")
    if raw:
        (anchor_dir / "subzones.md").write_text(raw, encoding="utf-8")

    for sz in (sub_resp.get("subzones") or [])[:4]:
        raw_sz = sz.get("polygon") or sz.get("polygon_px") or []
        poly = _normalize_or_pixel_polygon(raw_sz, mosaic.width, mosaic.height)
        geom = _clip_and_convert(poly, mosaic, image_ref="mosaic")
        if geom is None:
            continue
        subzone_geoms.append((FishableSubzone(
            subzone_id=sz.get("subzone_id") or f"{anchor_id}-s{len(subzone_geoms) + 1}",
            anchor_id=anchor_id,
            subzone_type=sz.get("subzone_type", "other"),
            geometry=geom,
            relative_priority=_clamp(float(sz.get("relative_priority", 0.5)), 0.0, 1.0),
            reasoning_summary=sz.get("reasoning_summary", ""),
            confidence=_clamp(float(sz.get("confidence", 0.5)), 0.0, 1.0),
        ), geom.pixel_polygon))

    # --- Typed assembly ---
    anchor_obj = AnchorStructure(
        anchor_id=anchor_id,
        structure_type=anchor.get("structure_type", "other"),
        scale=anchor.get("scale", "minor"),
        anchor_center_latlon=anchor_center_latlon,
        geometry=anchor_geom,
        orientation_deg=float(anchor["orientation_deg"])
            if anchor.get("orientation_deg") is not None else None,
        confidence=_clamp(float(anchor.get("confidence", 0.0)), 0.0, 1.0),
        source_images_used=[str(mosaic_path)],
        rationale=anchor.get("rationale", ""),
    )
    complex_obj = LocalComplex(
        complex_id=f"{anchor_id}-complex",
        anchor_id=anchor_id,
        member_features=list(complex_block.get("member_features", [])),
        relationship_summary=complex_block.get("relationship_summary", ""),
        geometry=complex_geom,
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
    for sz_obj, _poly in subzone_geoms:
        result.subzones.append(sz_obj)

    # --- Annotated PNG ---
    anchor_polys = [(anchor_obj.anchor_id, list(anchor_geom.pixel_polygon))]
    complex_polys = [(f"{anchor_id} cx", list(complex_geom.pixel_polygon))]
    influence_polys = [(f"{anchor_id} infl", list(influence_geom.pixel_polygon))]
    subzone_polys = [(sz.subzone_id, list(poly)) for sz, poly in subzone_geoms]

    annotated_path = anchor_dir / "annotated.png"
    render_annotated(
        base_image=mosaic.image,
        out_path=annotated_path,
        anchor_polygons_px=anchor_polys,
        complex_polygons_px=complex_polys,
        influence_polygons_px=influence_polys,
        subzone_polygons_px=subzone_polys,
    )
    result.annotated_image_paths[anchor_id] = str(annotated_path)
    result.mosaic_image_paths[anchor_id] = str(mosaic_path)

    # Per-anchor result.json
    per_anchor = {
        "anchor": anchor_obj.model_dump(),
        "local_complex": complex_obj.model_dump(),
        "influence_zone": influence_obj.model_dump(),
        "subzones": [sz.model_dump() for sz, _ in subzone_geoms],
    }
    (anchor_dir / "result.json").write_text(
        json.dumps(per_anchor, indent=2), encoding="utf-8",
    )


# --- Overlap audit ---


def _audit_overlaps(result: StructurePhaseResult) -> None:
    """Log all IoU > 0.1 overlaps; subordinate anchor pairs with IoU > 0.25."""
    if not result.anchors:
        return
    canvas = (TILE_PX * 5, TILE_PX * 5)  # big enough for any mosaic we build

    # Anchor vs anchor
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
            # Higher confidence wins; if tied, keep the first.
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

    # Strip influence zones of subordinated anchors
    if subordinated:
        result.influences = [
            inf for inf in result.influences if inf.anchor_id not in subordinated
        ]
        result.subordinated_ids.extend(subordinated)

    # Complex / influence / subzone overlaps — log only, no policy action
    _log_iou_overlaps(result, "complex", result.complexes, canvas)
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
            id_a = getattr(a, "anchor_id", None) or getattr(a, "complex_id", None) \
                or getattr(a, "influence_zone_id", None) or getattr(a, "subzone_id", "?")
            id_b = getattr(b, "anchor_id", None) or getattr(b, "complex_id", None) \
                or getattr(b, "influence_zone_id", None) or getattr(b, "subzone_id", "?")
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
    processed: list[str],
    deferred_entries: list[DeferredAnchor],
    truncated_ids: list[str],
    failed_geometry_ids: list[str],
    overlap_report: list[OverlapEntry],
    result: StructurePhaseResult,
) -> None:
    payload = {
        "cell_id": cell_id,
        "cell_center_latlon": list(cell_center),
        "discovered": discovered,
        "processed": processed,
        "deferred": [d.model_dump() for d in deferred_entries],
        "truncated": truncated_ids,
        "failed_geometry": failed_geometry_ids,
        "subordinated": list(result.subordinated_ids),
        "overlap_report": [o.model_dump() for o in overlap_report],
        "final_accepted": {
            "anchors": [a.anchor_id for a in result.anchors if a.anchor_id not in result.subordinated_ids],
            "complexes": [c.complex_id for c in result.complexes],
            "influences": [i.influence_zone_id for i in result.influences],
            "subzones": [s.subzone_id for s in result.subzones],
        },
        "api_calls_used": result.api_calls_used,
        "tiles_fetched": result.tiles_fetched,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Small geometry helper kept local to agent ---


def _anchor_center_latlon(
    bbox_px_z16: list[int] | tuple[int, int, int, int],
    z16_center: tuple[float, float],
) -> tuple[float, float]:
    """The center of an anchor's z16 bbox in lat/lon (used to ground resolve_continuation)."""
    from readwater.pipeline.structure.geo import pixel_to_latlon

    x, y, w, h = bbox_px_z16
    cx = x + w / 2.0
    cy = y + h / 2.0
    return pixel_to_latlon(
        cx, cy, Z16_CELL_PX, z16_center[0], z16_center[1], 16,
    )


# Re-export for tests and tooling.
__all__ = [
    "StructurePhase",
    "StructureBudget",
    "StructurePaths",
    "run_structure_phase",
]
