"""Structure-phase orchestrator (Phase 1).

State machine:
  DISCOVER -> RANK_AND_DEFER -> per anchor:
    PLAN_CAPTURE -> FETCH -> RESOLVE_CONTINUATION (loop-capped)
    -> IDENTIFY -> VALIDATE_SEEDS -> EXTRACT
    -> IDENTIFY_SUBZONES -> VALIDATE_SUBZONE_SEEDS -> EXTRACT_SUBZONES
    -> INTERPRET -> ASSEMBLE
  -> FINALIZE

Phase 1 uses ClickBoxExtractor for all extraction modes. Phase 2 replaces the
region (and later corridor, edge_band) extractors with smarter implementations
without touching this file. The separation is what makes that swap safe.
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
    ExtractorOutput,
    get_extractor,
    is_subzone_type_allowed,
    mode_for,
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
    default_excluded_regions,
    validate_seeds,
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
    VALIDATE_SEEDS = "validate_seeds"
    EXTRACT = "extract"
    IDENTIFY_SUBZONES = "identify_subzones"
    VALIDATE_SUBZONE_SEEDS = "validate_subzone_seeds"
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


def _frac_points_to_px(
    frac_points: list,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Convert a list of [x_frac, y_frac] in [0,1] to pixel integer tuples."""
    out: list[tuple[int, int]] = []
    for pt in frac_points or []:
        try:
            x = int(round(float(pt[0]) * width))
            y = int(round(float(pt[1]) * height))
        except (TypeError, ValueError, IndexError):
            continue
        out.append((x, y))
    return out


def _normalize_or_pixel_polygon(
    polygon: list,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Interpret a polygon as normalized [0-1] or pixel coords, return pixel."""
    if not polygon:
        return []
    flat = [v for pt in polygon for v in pt]
    if flat and max(abs(v) for v in flat) <= 1.5:
        return [
            (int(round(float(p[0]) * width)), int(round(float(p[1]) * height)))
            for p in polygon
        ]
    return [(int(round(p[0])), int(round(p[1]))) for p in polygon]


def _build_observed_geometry(
    extractor_output: ExtractorOutput,
    mode: str,
    mosaic: Mosaic,
    positive_points_px: list[tuple[int, int]],
    negative_points_px: list[tuple[int, int]],
    image_ref: str = "mosaic",
) -> ObservedGeometry | None:
    """Clip the extractor's polygon to the mosaic and attach lat/lon."""
    clipped = clip_polygon_to_rect(
        [(float(x), float(y)) for x, y in extractor_output.pixel_polygon],
        mosaic.width,
        mosaic.height,
    )
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)
    return ObservedGeometry(
        pixel_polygon=clipped,
        latlon_polygon=latlon,
        image_ref=image_ref,
        extractor=extractor_output.extractor_name,
        extraction_mode=mode,
        seed_positive_points=positive_points_px,
        seed_negative_points=negative_points_px,
        confidence=extractor_output.confidence,
    )


def _build_interpreted_geometry(
    polygon_px: list[tuple[int, int]],
    mosaic: Mosaic,
    source: str,
    rationale: str = "",
    image_ref: str = "mosaic",
) -> InterpretedGeometry | None:
    clipped = clip_polygon_to_rect(
        [(float(x), float(y)) for x, y in polygon_px],
        mosaic.width,
        mosaic.height,
    )
    if len(clipped) < 3:
        return None
    latlon = mosaic.polygon_px_to_latlon(clipped)
    return InterpretedGeometry(
        pixel_polygon=clipped,
        latlon_polygon=latlon,
        image_ref=image_ref,
        source=source,
        rationale=rationale,
    )


def _resolve_influence_polygon_px(
    influence_field: Any,
    anchor_polygon_px: list[tuple[int, int]],
    mosaic_width: int,
    mosaic_height: int,
) -> tuple[list[tuple[int, int]], str]:
    """Normalize the LLM's influence_zone polygon field. Returns (polygon_px, source)."""
    if isinstance(influence_field, dict) and influence_field.get("convex_hull_of_anchor"):
        hull = convex_hull(anchor_polygon_px)
        if "expand_frac" in influence_field:
            expand = int(
                round(
                    float(influence_field["expand_frac"])
                    * min(mosaic_width, mosaic_height),
                ),
            )
        else:
            expand = int(influence_field.get("expand_px", 100))
        return expand_polygon(hull, max(expand, 0)), "convex_hull_of_anchor"
    if isinstance(influence_field, list):
        return (
            _normalize_or_pixel_polygon(influence_field, mosaic_width, mosaic_height),
            "llm_polygon",
        )
    return [], "llm_polygon"


# --- Discovery preview image ---


def _draw_discovery_preview(
    z16_image_path: str,
    anchors: list[dict],
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


def _anchor_center_latlon(
    bbox_px_z16: list[int] | tuple[int, int, int, int],
    z16_center: tuple[float, float],
) -> tuple[float, float]:
    x, y, w, h = bbox_px_z16
    cx = x + w / 2.0
    cy = y + h / 2.0
    return pixel_to_latlon(
        cx, cy, Z16_CELL_PX, z16_center[0], z16_center[1], 16,
    )


# --- Seed validation + regenerate wrapper ---


async def _identify_with_retry(
    *,
    mosaic_image_path: str,
    anchor: dict,
    mosaic: Mosaic,
    identify_fn,
    get_points_to_validate,
    feedback_note_template: str = "",
    budget: StructureBudget,
) -> dict | None:
    """Run an identify call; validate; on failure, retry ONCE with a feedback note.

    `identify_fn(mosaic_path, anchor, feedback_note)` is either llm.identify_anchor
    or llm.identify_subzones. `get_points_to_validate(response)` returns a list of
    (feature_id, positives_px, negatives_px, mode) tuples to pass through the
    validator. If ANY feature fails, we regenerate the whole call once; the second
    attempt's per-feature failures are handled by the caller (drop).
    Returns the second-attempt response on retry, or None if the first call
    raises.
    """
    budget.charge_call()
    try:
        resp = await identify_fn(mosaic_image_path, anchor, "")
    except Exception as e:  # noqa: BLE001
        logger.warning("identify call failed: %s", e)
        return None

    # Run validation on first-pass response.
    issues = _collect_validation_issues(resp, mosaic, get_points_to_validate)
    if not issues:
        return resp

    # Regenerate once with feedback.
    feedback = "; ".join(f"{fid}: {reason}" for fid, reason in issues)
    if feedback_note_template:
        feedback = feedback_note_template.format(issues=feedback)
    logger.info("seed validation failed; regenerating once (%s)", feedback[:200])
    if budget.anchor_exhausted():
        return resp
    budget.charge_call()
    try:
        resp2 = await identify_fn(mosaic_image_path, anchor, feedback)
    except Exception as e:  # noqa: BLE001
        logger.warning("identify regenerate failed: %s", e)
        return resp
    return resp2


def _collect_validation_issues(
    resp: dict,
    mosaic: Mosaic,
    get_points_to_validate,
) -> list[tuple[str, str]]:
    """Run validator on every feature; return a list of (feature_id, reason) failures."""
    issues: list[tuple[str, str]] = []
    excluded = default_excluded_regions((mosaic.width, mosaic.height))
    for feature_id, pos_px, neg_px, mode in get_points_to_validate(resp, mosaic):
        r = validate_seeds(
            pos_px, neg_px,
            image_size=(mosaic.width, mosaic.height),
            extraction_mode=mode,
            excluded_regions=excluded,
        )
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
    extractor_registry: dict | None = None,
) -> StructurePhaseResult:
    """Run the full structure phase for one confirmed zoom-16 cell.

    `extractor_registry` (optional) overrides the default extractor-per-mode
    mapping. Tests and Phase 2+ use this to swap SAM in without agent changes.
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

    anchors_raw: list[dict] = list(discovery.get("anchors", []))
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

    # --- Per-anchor processing ---
    mosaic_state = MosaicState()
    for a, _rank in top:
        try:
            await _process_anchor(
                anchor=a,
                cell_id=cell_id,
                cell_center=cell_center,
                provider=provider,
                paths=paths,
                budget=budget,
                mosaic_state=mosaic_state,
                result=result,
                extractor_registry=extractor_registry,
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
    provider: ImageProvider,
    paths: StructurePaths,
    budget: StructureBudget,
    mosaic_state: MosaicState,
    result: StructurePhaseResult,
    extractor_registry: dict | None,
) -> None:
    anchor_id: str = anchor["anchor_id"]
    anchor_dir = paths.structures_root / anchor_id
    anchor_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = anchor_dir / "tiles"

    budget.start_anchor()

    # --- PLAN_CAPTURE (deterministic) ---
    logger.info("[structure:%s:%s] PLAN_CAPTURE", cell_id, anchor_id)
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

    # --- IDENTIFY + VALIDATE (anchor + member seeds + influence polygon) ---
    logger.info("[structure:%s:%s] IDENTIFY (anchor+members+influence)", cell_id, anchor_id)
    identify_resp = await _identify_with_retry(
        mosaic_image_path=str(mosaic_path),
        anchor=anchor,
        mosaic=mosaic,
        identify_fn=llm.identify_anchor,
        get_points_to_validate=_anchor_response_validation_points,
        budget=budget,
    )
    if identify_resp is None:
        result.truncated_ids.append(anchor_id)
        result.truncated = True
        return
    raw = identify_resp.pop("raw_response", "")
    if raw:
        (anchor_dir / "identify.md").write_text(raw, encoding="utf-8")

    # --- EXTRACT anchor ---
    anchor_block = identify_resp.get("anchor") or {}
    anchor_pos_px = _frac_points_to_px(
        anchor_block.get("positive_points"), mosaic.width, mosaic.height,
    )
    anchor_neg_px = _frac_points_to_px(
        anchor_block.get("negative_points"), mosaic.width, mosaic.height,
    )
    if not anchor_pos_px:
        _fail_identification(
            result, anchor_id, "anchor",
            "no valid positive points after validation/regeneration",
            regeneration_attempted=True,
        )
        return

    anchor_mode = mode_for(anchor.get("structure_type", "other"), "anchor")
    anchor_extractor = get_extractor(anchor_mode, extractor_registry)
    anchor_extract = anchor_extractor.extract(
        mosaic.image, anchor_pos_px, anchor_neg_px, bbox_hint=None,
    )
    anchor_geom = _build_observed_geometry(
        anchor_extract, anchor_mode, mosaic, anchor_pos_px, anchor_neg_px,
    )
    if anchor_geom is None:
        anchor_geom = _fallback_region_geometry(
            anchor_pos_px, anchor_neg_px, mosaic, result, anchor_id, "anchor",
            reason="primary anchor extractor returned degenerate polygon",
            primary_extractor=anchor_extract.extractor_name,
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
        m_pos = _frac_points_to_px(
            m.get("positive_points"), mosaic.width, mosaic.height,
        )
        m_neg = _frac_points_to_px(
            m.get("negative_points"), mosaic.width, mosaic.height,
        )
        if not m_pos:
            _fail_identification(
                result, m_id, "complex_member",
                "no positive points", regeneration_attempted=True,
            )
            continue
        m_mode = mode_for(m_type, "anchor")  # members use anchor vocabulary
        m_extractor = get_extractor(m_mode, extractor_registry)
        m_out = m_extractor.extract(mosaic.image, m_pos, m_neg)
        m_geom = _build_observed_geometry(m_out, m_mode, mosaic, m_pos, m_neg)
        if m_geom is None:
            m_geom = _fallback_region_geometry(
                m_pos, m_neg, mosaic, result, m_id, "complex_member",
                reason="member extractor returned degenerate polygon",
                primary_extractor=m_out.extractor_name,
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
    raw_poly = influence_block.get("polygon", [])
    influence_poly_px, influence_source = _resolve_influence_polygon_px(
        raw_poly, anchor_geom.pixel_polygon, mosaic.width, mosaic.height,
    )
    influence_geom = _build_interpreted_geometry(
        influence_poly_px, mosaic,
        source=influence_source,
        rationale=influence_block.get("rationale", ""),
    )
    if influence_geom is None:
        expand_default = int(round(0.10 * min(mosaic.width, mosaic.height)))
        fallback_poly = expand_polygon(
            convex_hull(anchor_geom.pixel_polygon), expand_default,
        )
        influence_geom = _build_interpreted_geometry(
            fallback_poly, mosaic, source="convex_hull_of_anchor",
            rationale="orchestrator fallback: LLM polygon collapsed",
        )
    # influence_geom should now be non-None; keep a defensive guard.
    if influence_geom is None:
        _fail_identification(
            result, anchor_id, "anchor",
            "influence polygon collapsed and hull fallback also failed",
            regeneration_attempted=False,
        )
        return

    # --- IDENTIFY_SUBZONES + VALIDATE + EXTRACT ---
    subzones_out: list[FishableSubzone] = []
    if not budget.anchor_exhausted():
        logger.info(
            "[structure:%s:%s] IDENTIFY_SUBZONES", cell_id, anchor_id,
        )
        subzone_resp = await _identify_with_retry(
            mosaic_image_path=str(mosaic_path),
            anchor=anchor,
            mosaic=mosaic,
            identify_fn=llm.identify_subzones,
            get_points_to_validate=_subzone_response_validation_points,
            budget=budget,
        )
        if subzone_resp is not None:
            raw_sz = subzone_resp.pop("raw_response", "")
            if raw_sz:
                (anchor_dir / "subzones.md").write_text(raw_sz, encoding="utf-8")
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
                sz_pos = _frac_points_to_px(
                    sz.get("positive_points"), mosaic.width, mosaic.height,
                )
                sz_neg = _frac_points_to_px(
                    sz.get("negative_points"), mosaic.width, mosaic.height,
                )
                if not sz_pos:
                    _fail_identification(
                        result, sz_id, "subzone",
                        "no positive points", regeneration_attempted=True,
                    )
                    continue
                sz_mode = mode_for(sz_type, "subzone")
                sz_extractor = get_extractor(sz_mode, extractor_registry)
                sz_out = sz_extractor.extract(mosaic.image, sz_pos, sz_neg)
                sz_geom = _build_observed_geometry(
                    sz_out, sz_mode, mosaic, sz_pos, sz_neg,
                )
                if sz_geom is None:
                    sz_geom = _fallback_region_geometry(
                        sz_pos, sz_neg, mosaic, result, sz_id, "subzone",
                        reason="subzone extractor returned degenerate polygon",
                        primary_extractor=sz_out.extractor_name,
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
        orientation_deg=float(anchor["orientation_deg"])
            if anchor.get("orientation_deg") is not None else None,
        confidence=_clamp(float(anchor.get("confidence", 0.0)), 0.0, 1.0),
        rationale=anchor.get("rationale", ""),
        source_images_used=[str(mosaic_path)],
    )
    complex_obj = LocalComplex(
        complex_id=f"{anchor_id}-complex",
        anchor_id=anchor_id,
        members=member_features,
        relationship_summary=complex_block.get("relationship_summary", ""),
        envelope=None,  # Phase 1: no envelope. Phase 3 can derive one.
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

    annotated_path = anchor_dir / "annotated.png"
    all_seed_points = _collect_all_seed_points(
        anchor_geom, member_features, subzones_out,
    )
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


# --- Validation-point extractors (per response schema) ---


def _anchor_response_validation_points(resp: dict, mosaic: Mosaic):
    """Yield (feature_id, positives_px, negatives_px, mode) for every feature in
    an identify_anchor response."""
    anchor_block = resp.get("anchor") or {}
    pos = _frac_points_to_px(
        anchor_block.get("positive_points"), mosaic.width, mosaic.height,
    )
    neg = _frac_points_to_px(
        anchor_block.get("negative_points"), mosaic.width, mosaic.height,
    )
    # Anchor mode depends on structure_type but we don't have it here; the agent
    # uses "region" as the validation mode (most permissive min-positives=1).
    yield ("anchor", pos, neg, "region")

    for idx, m in enumerate((resp.get("local_complex") or {}).get("members") or []):
        mpos = _frac_points_to_px(
            m.get("positive_points"), mosaic.width, mosaic.height,
        )
        mneg = _frac_points_to_px(
            m.get("negative_points"), mosaic.width, mosaic.height,
        )
        m_type = m.get("feature_type", "bar")
        mmode = mode_for(m_type, "anchor")
        yield (f"member_{idx + 1}", mpos, mneg, mmode)


def _subzone_response_validation_points(resp: dict, mosaic: Mosaic):
    for sz in resp.get("subzones") or []:
        sz_type = sz.get("subzone_type", "other")
        pos = _frac_points_to_px(
            sz.get("positive_points"), mosaic.width, mosaic.height,
        )
        neg = _frac_points_to_px(
            sz.get("negative_points"), mosaic.width, mosaic.height,
        )
        mode = mode_for(sz_type, "subzone")
        yield (sz.get("subzone_id", sz_type), pos, neg, mode)


# --- Fallbacks + logging ---


def _fallback_region_geometry(
    positives: list[tuple[int, int]],
    negatives: list[tuple[int, int]],
    mosaic: Mosaic,
    result: StructurePhaseResult,
    feature_id: str,
    feature_level: str,
    reason: str,
    primary_extractor: str,
) -> ObservedGeometry | None:
    """Use a raw ClickBoxExtractor(region) and tag it as a fallback."""
    fallback_extractor = ClickBoxExtractor(mode="region")
    out = fallback_extractor.extract(mosaic.image, positives, negatives)
    geom = _build_observed_geometry(
        out, "region", mosaic, positives, negatives,
    )
    if geom is None:
        return None
    # Overwrite the extractor name to make the fallback visible in artifacts.
    geom.extractor = "fallback"
    result.segmentation_issues.append(SegmentationIssue(
        feature_id=feature_id,
        feature_level=feature_level,
        extractor_attempted=primary_extractor,
        fallback_used="clickbox_region",
        reason=reason,
    ))
    return geom


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
    """Gather (label, positives, negatives) triples for rendering."""
    out: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]] = [
        ("anchor", anchor_geom.seed_positive_points, anchor_geom.seed_negative_points),
    ]
    for m in members:
        out.append((
            m.name,
            m.geometry.seed_positive_points,
            m.geometry.seed_negative_points,
        ))
    for sz in subzones:
        out.append((
            sz.subzone_id,
            sz.geometry.seed_positive_points,
            sz.geometry.seed_negative_points,
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
