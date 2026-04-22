"""Visual-context utilities and (later) the z16 handoff bundle assembler.

Phase 1 responsibilities:
  - Pure image-drawing: take an existing base image and write a single new
    PNG with one labeled rectangle. The base file is never mutated.
  - build_cell_context: LLM-backed wrapper that calls generate_cell_context,
    assigns deterministic IDs to the parsed payload, resolves local-idx
    references, and persists the raw response next to the cell image.
  - Digest helpers used to build the compact ancestor/scoring/thread blocks
    fed to the LLM.
  - (Step 9): assemble_z16_bundle / persist_bundle / load_bundle.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from readwater.api.claude_vision import generate_cell_context
from readwater.models.cell import BoundingBox
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
from readwater.pipeline.structure.geo import latlon_to_pixel

logger = logging.getLogger(__name__)

# --- Same-center bbox math (inlined to avoid a cell_analyzer import cycle) ---

_MILES_PER_DEG_LAT = 69.0
_EARTH_CIRCUMFERENCE_MILES = 24901.0


def _google_tile_miles(zoom: int, lat: float, image_size: int = 640) -> float:
    """Ground coverage in miles for a Google Static scale=2 tile.

    Mirrors pipeline.cell_analyzer.ground_coverage_miles but avoids importing
    from cell_analyzer (which imports this module).
    """
    tiles = image_size / 256
    return tiles * _EARTH_CIRCUMFERENCE_MILES * math.cos(math.radians(lat)) / (2**zoom)


def _same_center_bbox(center: tuple[float, float], zoom: int) -> BoundingBox:
    """Bbox of a same-center Google Static tile at the given zoom."""
    miles = _google_tile_miles(zoom, center[0])
    half_lat = (miles / 2) / _MILES_PER_DEG_LAT
    cos_lat = math.cos(math.radians(center[0])) or 1e-12
    half_lon = half_lat / cos_lat
    return BoundingBox(
        north=center[0] + half_lat,
        south=center[0] - half_lat,
        east=center[1] + half_lon,
        west=center[1] - half_lon,
    )

# Style constants — yellow stays visible over water, sand, and mangrove.
_OVERLAY_STROKE_RGB = (255, 221, 0)
_OVERLAY_STROKE_WIDTH = 6
_OVERLAY_LABEL_BG_RGB = (0, 0, 0)
_OVERLAY_LABEL_FG_RGB = (255, 221, 0)
_OVERLAY_LABEL_PAD = 6


def _try_load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font load; fall back to Pillow's bundled bitmap."""
    for candidate in ("DejaVuSans-Bold.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bbox_to_pixel_rect(
    inner_bbox: BoundingBox,
    base_center: tuple[float, float],
    base_zoom: int,
    image_size_px: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Project the inner bbox into pixel coords of the base image.

    Returns (x0, y0, x1, y1) in integer pixels with x0 <= x1 and y0 <= y1.
    Coordinates are clamped to the image rectangle; if the inner bbox falls
    entirely outside, the returned rect collapses to an image edge rather
    than raising.

    Base images in this codebase are always Google Static scale=2 square
    tiles (typically 1280x1280). The projection uses the image's square
    side length to match the convention of `structure.geo.latlon_to_pixel`.
    """
    w, h = image_size_px
    img_side = max(w, h)
    px_nw, py_nw = latlon_to_pixel(
        inner_bbox.north, inner_bbox.west,
        img_side, base_center[0], base_center[1], base_zoom,
    )
    px_se, py_se = latlon_to_pixel(
        inner_bbox.south, inner_bbox.east,
        img_side, base_center[0], base_center[1], base_zoom,
    )
    x0 = int(round(min(px_nw, px_se)))
    x1 = int(round(max(px_nw, px_se)))
    y0 = int(round(min(py_nw, py_se)))
    y1 = int(round(max(py_nw, py_se)))
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    return (x0, y0, x1, y1)


def draw_footprint_overlay(
    base_image_path: str | Path,
    base_center: tuple[float, float],
    base_zoom: int,
    inner_bbox: BoundingBox,
    out_path: str | Path,
    label: str = "",
) -> str:
    """Write a copy of the base image with `inner_bbox` drawn as a rectangle.

    The base image is expected to be a Google Static scale=2 tile centered
    on `base_center`. The base file is never modified. The output directory
    is created if missing.

    Args:
        base_image_path: existing on-disk image (PNG/JPEG).
        base_center: (lat, lon) center of the base image.
        base_zoom: Google Maps zoom of the base image.
        inner_bbox: geographic rectangle to outline on the base image.
        out_path: destination for the new PNG.
        label: optional text badge placed just above the rectangle.

    Returns:
        The output path as a string.
    """
    base_path = Path(base_image_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(base_path).convert("RGBA")
    x0, y0, x1, y1 = _bbox_to_pixel_rect(
        inner_bbox, base_center, base_zoom, img.size,
    )

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (x0, y0, x1, y1),
        outline=_OVERLAY_STROKE_RGB,
        width=_OVERLAY_STROKE_WIDTH,
    )

    if label:
        font = _try_load_font(size=max(18, img.size[1] // 48))
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            text_w, text_h = len(label) * 8, 16
        pad = _OVERLAY_LABEL_PAD
        lx = x0
        ly = max(0, y0 - text_h - 2 * pad)
        draw.rectangle(
            (lx, ly, lx + text_w + 2 * pad, ly + text_h + 2 * pad),
            fill=_OVERLAY_LABEL_BG_RGB,
        )
        draw.text(
            (lx + pad, ly + pad), label,
            fill=_OVERLAY_LABEL_FG_RGB, font=font,
        )

    overlay.convert("RGB").save(out, format="PNG")
    return str(out)


# ---------------------------------------------------------------------------
# Step 6 — build_cell_context + digest helpers
# ---------------------------------------------------------------------------


def _clamp01(x, default: float) -> float:
    """Best-effort coerce x to a float in [0, 1]; fall back to default."""
    if x is None:
        return default
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _digest_grid_scoring(grid_scoring_result: dict | None) -> str:
    """Compact digest of a dual-pass grid-scoring result for the LLM prompt.

    Includes the coarse triage summary, hydrology note, and kept/ambiguous/
    pruned counts. Returns "(none)" when nothing useful is available.
    """
    if not grid_scoring_result:
        return "(none)"
    parts: list[str] = []
    summary = (grid_scoring_result.get("summary") or "").strip()
    if summary:
        parts.append(f"summary: {summary}")
    hydro = (grid_scoring_result.get("hydrology_notes") or "").strip()
    if hydro:
        parts.append(f"hydrology: {hydro}")
    sub = grid_scoring_result.get("sub_scores") or []
    if sub:
        counts = {5: 0, 3: 0, 0: 0, "other": 0}
        for sc in sub:
            try:
                si = int(round(float(sc.get("score"))))
            except (TypeError, ValueError):
                counts["other"] += 1
                continue
            if si in (0, 3, 5):
                counts[si] += 1
            else:
                counts["other"] += 1
        parts.append(
            f"scores: kept={counts[5]}, ambiguous={counts[3]}, "
            f"pruned={counts[0]}, other={counts['other']}"
        )
    return "; ".join(parts) if parts else "(none)"


def _digest_ancestors(
    ancestor_lineage: list[LineageRef],
    ancestor_contexts: dict[str, CellContext],
) -> str:
    """Short chain-block of ancestor contexts for the LLM prompt.

    One header line per ancestor plus up to 3 top observations and 2 top
    morphology entries each. Bounded to keep prompt size predictable.
    """
    if not ancestor_lineage:
        return "(none)"
    lines: list[str] = []
    for ref in ancestor_lineage:
        ctx = ancestor_contexts.get(ref.cell_id)
        header = f"z{ref.zoom} {ref.cell_id}"
        if ctx is None:
            lines.append(f"{header}: (no context)")
            continue
        stats = (
            f"obs={len(ctx.observations)}, "
            f"morph={len(ctx.morphology)}, "
            f"threads={len(ctx.feature_threads)}"
        )
        lines.append(f"{header}: {stats}")
        for obs in ctx.observations[:3]:
            loc = f" [{obs.location_hint}]" if obs.location_hint else ""
            lines.append(f"  obs: {obs.label}{loc}")
        for m in ctx.morphology[:2]:
            lines.append(f"  morph ({m.kind}): {m.statement}")
    return "\n".join(lines)


def _collect_open_threads(
    ancestor_contexts: dict[str, CellContext],
    current_zoom: int,
) -> list[CandidateFeatureThread]:
    """Threads from ancestors that could be resolved at this zoom or deeper.

    Selection rule: status == 'hypothesized' and (needs_zoom is None
    or needs_zoom >= current_zoom). needs_zoom == None means "any
    deeper zoom" and is always included.
    """
    out: list[CandidateFeatureThread] = []
    for ctx in ancestor_contexts.values():
        for t in ctx.feature_threads:
            if t.status != "hypothesized":
                continue
            if t.needs_zoom is not None and t.needs_zoom < current_zoom:
                continue
            out.append(t)
    return out


def _format_open_threads(threads: list[CandidateFeatureThread]) -> str:
    """Render the open-thread block for the LLM prompt."""
    if not threads:
        return "(none)"
    return "\n".join(
        f"- {t.thread_id} ({t.feature_type}): {t.summary}"
        for t in threads
    )


def _parse_observations(cell_id: str, raw: list | None) -> list[DirectObservation]:
    out: list[DirectObservation] = []
    for i, o in enumerate(raw or []):
        if not isinstance(o, dict):
            continue
        label = (o.get("label") or "").strip()
        if not label:
            continue
        out.append(DirectObservation(
            observation_id=f"{cell_id}:obs:{i}",
            label=label,
            location_hint=(o.get("location_hint") or "").strip(),
            confidence=_clamp01(o.get("confidence"), 0.5),
        ))
    return out


def _parse_morphology(cell_id: str, raw: list | None) -> list[MorphologyInference]:
    out: list[MorphologyInference] = []
    for i, m in enumerate(raw or []):
        if not isinstance(m, dict):
            continue
        kind = (m.get("kind") or "").strip()
        stmt = (m.get("statement") or "").strip()
        if not (kind and stmt):
            continue
        refs_raw = m.get("references") or []
        refs = [str(r) for r in refs_raw if r is not None]
        out.append(MorphologyInference(
            inference_id=f"{cell_id}:morph:{i}",
            kind=kind,
            statement=stmt,
            references=refs,
            confidence=_clamp01(m.get("confidence"), 0.5),
        ))
    return out


def _parse_threads(
    cell_id: str,
    raw: list | None,
    observations: list[DirectObservation],
    open_thread_ids: set[str],
) -> list[CandidateFeatureThread]:
    out: list[CandidateFeatureThread] = []
    obs_count = len(observations)
    for i, t in enumerate(raw or []):
        if not isinstance(t, dict):
            continue
        ftype = (t.get("feature_type") or "").strip()
        status = (t.get("status") or "").strip()
        if not (ftype and status):
            continue
        idxs = t.get("supporting_observations_local_idx") or []
        supp: list[str] = []
        for idx in idxs:
            try:
                ii = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < obs_count:
                supp.append(observations[ii].observation_id)
        parent = t.get("parent_thread_id")
        if parent is not None and parent not in open_thread_ids:
            parent = None  # drop dangling references
        needs = t.get("needs_zoom")
        try:
            needs_int = int(needs) if needs is not None else None
        except (TypeError, ValueError):
            needs_int = None
        out.append(CandidateFeatureThread(
            thread_id=f"{cell_id}:th:{i}",
            feature_type=ftype,
            status=status,
            summary=(t.get("summary") or "").strip(),
            supporting_observation_ids=supp,
            parent_thread_id=parent,
            needs_zoom=needs_int,
            confidence=_clamp01(t.get("confidence"), 0.3),
        ))
    return out


def _parse_questions(cell_id: str, raw: list | None) -> list[UnresolvedQuestion]:
    out: list[UnresolvedQuestion] = []
    for i, q in enumerate(raw or []):
        if not isinstance(q, dict):
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        tz = q.get("target_zoom")
        try:
            tz_int = int(tz) if tz is not None else None
        except (TypeError, ValueError):
            tz_int = None
        out.append(UnresolvedQuestion(
            question_id=f"{cell_id}:q:{i}",
            question=text,
            target_zoom=tz_int,
        ))
    return out


async def build_cell_context(
    cell_id: str,
    zoom: int,
    image_path: str,
    center: tuple[float, float],
    coverage_miles: float,
    ancestor_lineage: list[LineageRef] | None = None,
    ancestor_contexts: dict[str, CellContext] | None = None,
    grid_scoring_result: dict | None = None,
    model_used: str = "",
    raw_response_dir: Path | str | None = None,
) -> CellContext:
    """Produce a validated CellContext for a retained cell.

    Calls generate_cell_context, assigns deterministic IDs, resolves
    local-idx supporting-observation references, drops dangling
    parent_thread_id references, and writes the raw LLM markdown next
    to the cell image (unless raw_response_dir is given).

    Returns a validated CellContext even when the model produced an empty
    or partially malformed response (malformed entries are dropped).
    """
    ancestor_lineage = ancestor_lineage or []
    ancestor_contexts = ancestor_contexts or {}

    ancestor_block = _digest_ancestors(ancestor_lineage, ancestor_contexts)
    grid_digest = _digest_grid_scoring(grid_scoring_result)
    open_threads = _collect_open_threads(ancestor_contexts, zoom)
    open_thread_block = _format_open_threads(open_threads)
    open_thread_ids = {t.thread_id for t in open_threads}

    parsed = await generate_cell_context(
        image_path=image_path,
        cell_id=cell_id,
        zoom=zoom,
        center=center,
        coverage_miles=coverage_miles,
        ancestor_chain_block=ancestor_block,
        grid_scoring_digest=grid_digest,
        open_thread_block=open_thread_block,
    )

    raw_text = parsed.pop("raw_response", "") if isinstance(parsed, dict) else ""
    observations = _parse_observations(cell_id, parsed.get("observations") if isinstance(parsed, dict) else None)
    morphology = _parse_morphology(cell_id, parsed.get("morphology") if isinstance(parsed, dict) else None)
    threads = _parse_threads(
        cell_id,
        parsed.get("feature_threads") if isinstance(parsed, dict) else None,
        observations,
        open_thread_ids,
    )
    questions = _parse_questions(cell_id, parsed.get("open_questions") if isinstance(parsed, dict) else None)

    raw_path: str | None = None
    if raw_text:
        src_image = Path(image_path)
        target_dir = Path(raw_response_dir) if raw_response_dir else src_image.parent
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            raw_file = target_dir / f"{src_image.stem}_context.md"
            raw_file.write_text(raw_text, encoding="utf-8")
            raw_path = str(raw_file)
        except OSError as e:
            logger.warning("Failed to persist cell-context raw response: %s", e)

    return CellContext(
        cell_id=cell_id,
        zoom=zoom,
        observations=observations,
        morphology=morphology,
        feature_threads=threads,
        open_questions=questions,
        evidence=[],
        model_used=model_used,
        source_images=[image_path],
        raw_response_path=raw_path,
    )


# ---------------------------------------------------------------------------
# Step 9 — Z16ContextBundle assembly + persistence
# ---------------------------------------------------------------------------


def _structure_dir_for(base_output_dir: Path | str, cell_id: str) -> Path:
    """Per-cell structures/{cell_id}/ directory (matches run_structure_phase)."""
    return Path(base_output_dir) / "structures" / cell_id


def _role_from_zoom_delta(self_zoom: int, ancestor_zoom: int) -> VisualRole | None:
    """Map (self.zoom - ancestor.zoom) to the appropriate ancestor visual role.

    Returns None for unknown deltas. The supported deltas match the recursion
    step of +2 zoom levels per depth.
    """
    delta = self_zoom - ancestor_zoom
    if delta == 2:
        return VisualRole.Z14_PARENT
    if delta == 4:
        return VisualRole.Z12_GRANDPARENT
    if delta == 6:
        return VisualRole.Z10_GREAT_GRANDPARENT
    return None


def _footprint_label(inner: LineageRef, is_self: bool) -> str:
    """Return the overlay_draws label for an inner-footprint overlay."""
    zoom = inner.zoom
    # Labels use z{zoom}_footprint so consumers can distinguish what the box
    # represents regardless of where it sits in the chain.
    return f"z{zoom}_footprint" if not is_self else "z16_footprint"


def assemble_z16_bundle(
    self_lineage: LineageRef,
    self_context: CellContext,
    ancestor_lineage: list[LineageRef],
    ancestor_contexts: dict[str, CellContext],
    z15_same_center_path: str | None,
    base_output_dir: Path | str,
) -> Z16ContextBundle:
    """Compile a Z16ContextBundle from recursion state.

    Generates overlay files on disk — one per recursive ancestor plus one
    for z15_same_center — into {base_output_dir}/structures/{cell_id}/.
    Base images are never mutated. Returns the in-memory bundle; the caller
    is responsible for persistence via persist_bundle.

    z15_same_center_path is expected to be the context image that was
    already fetched during grid scoring at the z16 cell. Pass None if that
    image is not available; the bundle will simply omit Z15_SAME_CENTER.
    """
    cell_id = self_lineage.cell_id
    structures_dir = _structure_dir_for(base_output_dir, cell_id)
    structures_dir.mkdir(parents=True, exist_ok=True)

    full_lineage: list[LineageRef] = [*ancestor_lineage, self_lineage]
    contexts: dict[str, CellContext] = {
        **ancestor_contexts,
        cell_id: self_context,
    }
    evidence: dict[str, list[EvidenceSummary]] = {
        ref.cell_id: contexts[ref.cell_id].evidence
        for ref in full_lineage
        if ref.cell_id in contexts
    }

    visuals: dict[VisualRole, VisualContextRef] = {}

    # z16_local — no overlay.
    visuals[VisualRole.Z16_LOCAL] = VisualContextRef(
        role=VisualRole.Z16_LOCAL,
        zoom=self_lineage.zoom,
        center=self_lineage.center,
        depicts_bbox=self_lineage.bbox,
        base_image_path=self_lineage.image_path or "",
        lineage_cell_id=self_lineage.cell_id,
    )

    # z15_same_center — overlay z16 footprint.
    if z15_same_center_path:
        z15_overlay = structures_dir / "overlay_z15_same_center.png"
        try:
            draw_footprint_overlay(
                base_image_path=z15_same_center_path,
                base_center=self_lineage.center,
                base_zoom=15,
                inner_bbox=self_lineage.bbox,
                out_path=str(z15_overlay),
                label=self_lineage.cell_id,
            )
            overlay_str: str | None = str(z15_overlay)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to render z15_same_center overlay for %s: %s",
                cell_id, exc,
            )
            overlay_str = None
        visuals[VisualRole.Z15_SAME_CENTER] = VisualContextRef(
            role=VisualRole.Z15_SAME_CENTER,
            zoom=15,
            center=self_lineage.center,
            depicts_bbox=_same_center_bbox(self_lineage.center, 15),
            overlay_footprint_bbox=self_lineage.bbox,
            base_image_path=z15_same_center_path,
            overlay_image_path=overlay_str,
            overlay_draws=["z16_footprint"],
            lineage_cell_id=None,
        )

    # Recursive ancestor visuals — each overlay draws the NEXT-INNER retained bbox.
    for i, anc in enumerate(ancestor_lineage):
        role = _role_from_zoom_delta(self_lineage.zoom, anc.zoom)
        if role is None:
            continue
        next_inner: LineageRef = (
            ancestor_lineage[i + 1] if i + 1 < len(ancestor_lineage) else self_lineage
        )
        overlay_path = structures_dir / f"overlay_{role.value}.png"
        overlay_str = None
        if anc.image_path:
            try:
                draw_footprint_overlay(
                    base_image_path=anc.image_path,
                    base_center=anc.center,
                    base_zoom=anc.zoom,
                    inner_bbox=next_inner.bbox,
                    out_path=str(overlay_path),
                    label=next_inner.cell_id,
                )
                overlay_str = str(overlay_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to render overlay for %s on %s: %s",
                    role.value, anc.cell_id, exc,
                )
        visuals[role] = VisualContextRef(
            role=role,
            zoom=anc.zoom,
            center=anc.center,
            depicts_bbox=anc.bbox,
            overlay_footprint_bbox=next_inner.bbox,
            base_image_path=anc.image_path or "",
            overlay_image_path=overlay_str,
            overlay_draws=[_footprint_label(next_inner, is_self=next_inner is self_lineage)],
            lineage_cell_id=anc.cell_id,
        )

    compiled_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return Z16ContextBundle(
        cell_id=cell_id,
        compiled_at=compiled_at,
        lineage=full_lineage,
        visuals=visuals,
        contexts=contexts,
        evidence=evidence,
    )


def persist_bundle(bundle: Z16ContextBundle, path: Path | str) -> str:
    """Write the bundle to disk as JSON. Returns the path as a string."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return str(p)


def bundle_path_for(base_output_dir: Path | str, cell_id: str) -> Path:
    """Canonical on-disk path for a z16 cell's context bundle JSON."""
    return _structure_dir_for(base_output_dir, cell_id) / "context_bundle.json"


def load_bundle(path: Path | str) -> Z16ContextBundle:
    """Read a persisted Z16ContextBundle from disk.

    Phase 1 defines this as the forward-compatible seam: any later phase
    that wires the structure pipeline to consume the bundle does so by
    calling this function with the path from `bundle_path_for(...)` (also
    recorded in metadata.json under `context_bundle_path`). Phase 1 does
    not modify src/readwater/pipeline/structure/agent.py at all.
    """
    return Z16ContextBundle.model_validate_json(
        Path(path).read_text(encoding="utf-8"),
    )
