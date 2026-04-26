"""Phase C v1 anchor discovery — config-driven DISCOVER + COORDS + PLAN_CAPTURE.

Replaces the legacy z16 grid-cell DISCOVER path (currently in
`structure/agent.py`). Produces `AnchorStructure[]` with populated
lat/lon centers, draft state, full provenance, phase history, findings,
and seed Z18FetchPlans for downstream Phase D capture.

Pipeline:

    DISCOVER  ─ v3 prompts (grid | nogrid | comparison)
       │
       ▼
    COORDS    ─ coord-gen prompts (grid | nogrid | comparison)
       │            per-anchor failure → keep + Finding(warn)
       │            whole-batch malformed JSON → raise
       ▼
    PLAN_CAPTURE ─ mosaic.z18_tile_plan_from_latlon()
       │
       ▼
    ASSEMBLE  ─ AnchorStructure[] with state="draft", real Provenance,
                3-event PhaseHistory, optional Findings, seed FetchPlan

See `docs/PHASE_C_DISCOVERY_PIPELINE.md` for the full design rationale.
The locked failure / matching policies come from the addendum in
`docs/PHASE_C_TASKS.md`:
  - Strict anchor_id matching (no fuzzy fallback).
  - Per-anchor coord failures (out-of-bounds, low confidence) keep the
    anchor with a Finding; seed_z18_fetch_plan stays None.
  - Whole-batch coord-gen JSON malformed → raise CoordsBatchFailure.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from PIL import Image as PILImage

from readwater.api.claude_vision import (
    MAX_TOKENS,
    MODEL,
    _extract_json_from_response,
    _get_client,
    _load_prompt,
)
from readwater.models.context import Z16ContextBundle
from readwater.models.structure import (
    AnchorStructure,
    Finding,
    ObservedGeometry,
    PhaseEvent,
    Provenance,
    StructurePhaseResult,
    Z18FetchPlan,
)
from readwater.pipeline.evidence import build_cell_evidence_section
from readwater.pipeline.structure.geo import (
    pixel_to_latlon,
    polygon_px_to_latlon,
)
from readwater.pipeline.structure.grid_overlay import (
    draw_label_grid,
    grid_shape_for_image,
)
from readwater.pipeline.structure.mosaic import z18_tile_plan_from_latlon

logger = logging.getLogger(__name__)

IMG_SIZE_PX = 1280
ZOOM = 16


# ----------------------------------------------------------------------
# Config + inputs
# ----------------------------------------------------------------------


V3Mode = Literal["nogrid", "grid", "comparison"]
CoordsMode = Literal["nogrid", "grid", "comparison"]
WinnerMode = Literal["nogrid", "grid"]


@dataclass(frozen=True)
class AnchorDiscoveryConfig:
    """Pipeline configuration. See docs/PHASE_C_DISCOVERY_PIPELINE.md."""

    v3_mode: V3Mode = "nogrid"
    coords_mode: CoordsMode = "grid"
    v3_comparison_winner: WinnerMode = "nogrid"
    coords_comparison_winner: WinnerMode = "grid"
    inject_evidence: bool = False
    tile_budget_z18: int = 25
    # Render a review PNG (z16 image + pipeline anchor bboxes/centers/labels,
    # optionally GT in red when gt_anchors.json sits next to the bundle).
    # Saved to <output_dir>/<cell_id>_review_overlay.png. Requires an
    # output_dir to be passed to run_anchor_discovery; no-op otherwise.
    render_review_overlay: bool = True


@dataclass
class AnchorDiscoveryInputs:
    """Per-cell inputs needed for the discovery pipeline.

    Constructed by the agent from its own state. Lives outside the config
    because it changes per-cell while config is stable across cells.
    """

    cell_id: str
    bundle: Z16ContextBundle
    z16_image_path: Path
    z16_grid_overlay_path: Path  # cached 8x8 overlay; rendered if missing
    overlay_z15_path: Path
    overlay_z14_path: Path
    overlay_z12_path: Path
    z16_center_latlon: tuple[float, float]
    coverage_miles: float = 0.37
    area_root: Path | None = None  # for evidence mask discovery; None disables auto-discovery


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class CoordsBatchFailure(RuntimeError):
    """Raised when coord-gen returns wholly unparseable JSON (no recovery)."""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_block(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _b64_png(path),
        },
    }


def ensure_grid_overlay(z16_image: Path, cache_path: Path) -> Path:
    """Render the 8×8 A1–H8 grid overlay on `z16_image`, cached at `cache_path`.

    Validates by IMAGE DIMENSIONS, not just mtime: a previous codebase
    iteration shipped a 4×4 numbered grid at the same legacy path, and a
    pure mtime check let it slip through. We require the cached image to
    have the same dimensions as the source AND a non-zero size.
    """
    if not z16_image.exists():
        raise FileNotFoundError(f"missing z16 source image: {z16_image}")

    with PILImage.open(z16_image) as src:
        src_size = src.size

    cache_ok = False
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with PILImage.open(cache_path) as cached:
                cache_ok = cached.size == src_size
        except Exception:  # noqa: BLE001 — corrupt cache → regenerate
            cache_ok = False
    if cache_ok:
        return cache_path

    rows, cols = grid_shape_for_image(src_size, short_axis_cells=8)
    if (rows, cols) != (8, 8):
        raise ValueError(
            f"expected (8,8) grid for {z16_image.name} but got ({rows},{cols}); "
            f"source image dims = {src_size}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    draw_label_grid(str(z16_image), rows, cols, str(cache_path))
    return cache_path


def _input_hash(*chunks: bytes | str) -> str:
    h = hashlib.sha256()
    for c in chunks:
        if isinstance(c, str):
            h.update(c.encode("utf-8"))
        else:
            h.update(c)
    return h.hexdigest()[:16]  # short hex for storage


def _provider_config() -> dict:
    return {"model": MODEL, "max_tokens": MAX_TOKENS}


# ----------------------------------------------------------------------
# DISCOVER stage
# ----------------------------------------------------------------------


async def discover_anchors_v3(
    inputs: AnchorDiscoveryInputs,
    file_mode: Literal["nogrid", "grid"],
    inject_evidence: bool,
) -> tuple[dict, str, dict]:
    """Run one v3 variant. Returns (parsed_json, raw_text, run_meta).

    `run_meta` records what was sent (image paths, prompt versions, hash) so
    callers can persist provenance without re-deriving anything.
    """
    if file_mode not in ("nogrid", "grid"):
        raise ValueError(f"file_mode must be nogrid|grid, got {file_mode!r}")

    system_prompt = _load_prompt(f"anchor_identification_v3_{file_mode}_system.txt")
    user_template = _load_prompt(f"anchor_identification_v3_{file_mode}_user.txt")

    bundle_json = json.dumps(inputs.bundle.model_dump(), indent=2, default=str)

    if inject_evidence and inputs.area_root is not None:
        evidence_table = build_cell_evidence_section(
            inputs.cell_id, inputs.area_root,
            grid_rows=8, grid_cols=8,
            image_size=(IMG_SIZE_PX, IMG_SIZE_PX),
        )
    elif inject_evidence:
        evidence_table = "(habitat evidence requested but no area_root provided)"
    else:
        evidence_table = "(habitat evidence injection disabled for this run)"

    user_prompt = user_template.format(
        cell_id=inputs.cell_id,
        zoom=ZOOM,
        center_lat=f"{inputs.z16_center_latlon[0]:.4f}",
        center_lon=f"{inputs.z16_center_latlon[1]:.4f}",
        coverage_miles=f"{inputs.coverage_miles:.2f}",
        context_bundle_json=bundle_json,
        evidence_table=evidence_table,
    )

    if file_mode == "grid":
        z16_path = ensure_grid_overlay(inputs.z16_image_path, inputs.z16_grid_overlay_path)
        image1_label = "IMAGE 1 — z16_local with 8x8 A1-H8 grid overlay (1280x1280):"
    else:
        z16_path = inputs.z16_image_path
        image1_label = "IMAGE 1 — z16_local clean image (1280x1280, no overlay):"

    content = [
        {"type": "text", "text": image1_label},
        _image_block(z16_path),
        {"type": "text", "text": "IMAGE 2 — z15_same_center (yellow = z16 footprint):"},
        _image_block(inputs.overlay_z15_path),
        {"type": "text", "text": "IMAGE 3 — z14_parent (yellow = z16 footprint inside z14):"},
        _image_block(inputs.overlay_z14_path),
        {"type": "text", "text": "IMAGE 4 — z12_grandparent (yellow = z14 footprint inside z12):"},
        _image_block(inputs.overlay_z12_path),
        {"type": "text", "text": user_prompt},
    ]

    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw_text = response.content[0].text
    try:
        parsed = _extract_json_from_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("v3 (%s) returned unparseable JSON: %s", file_mode, exc)
        parsed = {"anchors": []}

    run_meta = {
        "stage": "DISCOVER",
        "file_mode": file_mode,
        "inject_evidence": inject_evidence,
        "z16_image_used": str(z16_path),
        "z15_image_used": str(inputs.overlay_z15_path),
        "z14_image_used": str(inputs.overlay_z14_path),
        "z12_image_used": str(inputs.overlay_z12_path),
        "prompt_id": "anchor_identification_v3",
        "prompt_version": f"v3_{file_mode}",
        "input_hash": _input_hash(
            z16_path.read_bytes(),
            system_prompt,
            user_prompt,
        ),
        "ts": _utc_iso(),
    }
    return parsed, raw_text, run_meta


# ----------------------------------------------------------------------
# COORDS stage
# ----------------------------------------------------------------------


async def locate_anchor_coords(
    inputs: AnchorDiscoveryInputs,
    v3_anchors: list[dict],
    file_mode: Literal["nogrid", "grid"],
) -> tuple[dict, str, dict]:
    """Run one coord-gen variant against `v3_anchors`. Raises CoordsBatchFailure
    if the model returns wholly unparseable JSON."""
    if file_mode not in ("nogrid", "grid"):
        raise ValueError(f"file_mode must be nogrid|grid, got {file_mode!r}")

    system_prompt = _load_prompt(f"anchor_coords_{file_mode}_system.txt")
    user_template = _load_prompt(f"anchor_coords_{file_mode}_user.txt")

    if file_mode == "grid":
        coord_image = ensure_grid_overlay(
            inputs.z16_image_path, inputs.z16_grid_overlay_path,
        )
        image_label = "IMAGE 1 — z16_local with 8x8 A1-H8 grid overlay (1280x1280):"
    else:
        coord_image = inputs.z16_image_path
        image_label = "IMAGE 1 — z16_local clean image (1280x1280, no overlay):"

    user_prompt = user_template.format(
        cell_id=inputs.cell_id,
        z16_center_lat=f"{inputs.z16_center_latlon[0]:.6f}",
        z16_center_lon=f"{inputs.z16_center_latlon[1]:.6f}",
        anchor_list_json=json.dumps(v3_anchors, indent=2),
    )

    content = [
        {"type": "text", "text": image_label},
        _image_block(coord_image),
        {"type": "text", "text": user_prompt},
    ]

    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw_text = response.content[0].text
    try:
        parsed = _extract_json_from_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        # Per addendum: malformed batch JSON is a stage failure. Bubble up.
        raise CoordsBatchFailure(
            f"coord-gen ({file_mode}) returned unparseable JSON: {exc}"
        ) from exc
    if not isinstance(parsed.get("anchors"), list):
        raise CoordsBatchFailure(
            f"coord-gen ({file_mode}) JSON has no anchors[] list: {parsed!r}"
        )

    run_meta = {
        "stage": "COORDS",
        "file_mode": file_mode,
        "image_used": str(coord_image),
        "prompt_id": "anchor_coords",
        "prompt_version": f"coords_{file_mode}",
        "input_hash": _input_hash(
            coord_image.read_bytes(),
            system_prompt,
            user_prompt,
        ),
        "ts": _utc_iso(),
    }
    return parsed, raw_text, run_meta


# ----------------------------------------------------------------------
# PLAN_CAPTURE stage
# ----------------------------------------------------------------------


def _bbox_extent_meters(
    pixel_bbox: tuple[int, int, int, int], lat: float,
) -> float:
    """Convert a pixel bbox to a rough side-length in meters at this lat."""
    from readwater.pipeline.structure.geo import meters_per_pixel
    x0, y0, x1, y1 = pixel_bbox
    side_px = max(x1 - x0, y1 - y0)
    return side_px * meters_per_pixel(ZOOM, lat)


# ----------------------------------------------------------------------
# ASSEMBLE
# ----------------------------------------------------------------------


def _build_provenance(
    inputs: AnchorDiscoveryInputs,
    discover_meta: dict,
    coords_meta: dict,
) -> Provenance:
    overlay_refs: list[str] = []
    if discover_meta.get("file_mode") == "grid":
        overlay_refs.append(str(inputs.z16_grid_overlay_path))
    if coords_meta.get("file_mode") == "grid":
        gp = str(inputs.z16_grid_overlay_path)
        if gp not in overlay_refs:
            overlay_refs.append(gp)
    return Provenance(
        source_images=[
            discover_meta["z16_image_used"],
            discover_meta["z15_image_used"],
            discover_meta["z14_image_used"],
            discover_meta["z12_image_used"],
            coords_meta["image_used"],
        ],
        overlay_refs=overlay_refs,
        prompt_id=discover_meta["prompt_id"],
        prompt_version=f"{discover_meta['prompt_version']}+{coords_meta['prompt_version']}",
        provider_config=_provider_config(),
        input_hash=_input_hash(discover_meta["input_hash"], coords_meta["input_hash"]),
    )


def _coord_bbox_to_geometry(
    pixel_bbox: tuple[int, int, int, int],
    z16_center_latlon: tuple[float, float],
    image_ref: str,
) -> ObservedGeometry:
    x0, y0, x1, y1 = pixel_bbox
    pixel_polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    latlon_polygon = polygon_px_to_latlon(
        [(float(x), float(y)) for (x, y) in pixel_polygon],
        IMG_SIZE_PX, z16_center_latlon[0], z16_center_latlon[1], ZOOM,
    )
    return ObservedGeometry(
        pixel_polygon=pixel_polygon,
        latlon_polygon=latlon_polygon,
        image_ref=image_ref,
        extractor="coord_gen",
        extraction_mode="point_feature",
        seed_cells=[],
    )


def assemble_anchor_structures(
    inputs: AnchorDiscoveryInputs,
    config: AnchorDiscoveryConfig,
    v3_anchors: list[dict],
    coord_anchors: list[dict],
    discover_meta: dict,
    coords_meta: dict,
) -> tuple[list[AnchorStructure], list[Finding]]:
    """Stitch v3 + coord-gen + plan-capture into AnchorStructure[].

    Returns (anchors, cell_level_findings). Per-anchor findings live on each
    AnchorStructure.findings. `cell_level_findings` covers issues that don't
    attach to a single anchor (e.g. unmatched coord-gen entries).
    """
    cg_by_id = {a.get("anchor_id"): a for a in coord_anchors if a.get("anchor_id")}
    v3_id_set = {a.get("anchor_id") for a in v3_anchors}
    cell_findings: list[Finding] = []

    # Surface coord-gen entries that reference anchors v3 didn't emit.
    unmatched_cg = [aid for aid in cg_by_id.keys() if aid not in v3_id_set]
    for aid in unmatched_cg:
        cell_findings.append(Finding(
            issue_code="COORDS_UNMATCHED_ANCHOR_ID",
            severity="warn",
            object_id=aid,
            message=f"coord-gen returned anchor_id {aid!r} that v3 did not emit; ignored.",
        ))

    discover_event = PhaseEvent(
        phase="C.DISCOVER", action="emit",
        actor=f"anchor_discovery.v3_{discover_meta['file_mode']}",
        timestamp=discover_meta["ts"],
        note=f"v3 emitted {len(v3_anchors)} anchors",
    )
    coords_actor = f"anchor_discovery.coords_{coords_meta['file_mode']}"

    structures: list[AnchorStructure] = []
    for rank, v3a in enumerate(v3_anchors, start=1):
        aid = v3a.get("anchor_id")
        if not aid:
            cell_findings.append(Finding(
                issue_code="V3_MISSING_ANCHOR_ID",
                severity="warn",
                object_id="(unknown)",
                message="v3 emitted an anchor with no anchor_id; skipped.",
            ))
            continue

        per_findings: list[Finding] = []
        cg = cg_by_id.get(aid)
        pixel_center: tuple[float, float] | None = None
        pixel_bbox: tuple[int, int, int, int] | None = None
        placement_conf: float | None = None
        coords_note = "no coord-gen response"

        if cg is None:
            per_findings.append(Finding(
                issue_code="NO_COORD_RESPONSE",
                severity="warn",
                object_id=aid,
                message="coord-gen returned no entry for this anchor_id",
                recommended_action="rerun coord-gen or accept anchor without placement",
            ))
        else:
            try:
                pc = cg.get("pixel_center") or [None, None]
                pb = cg.get("pixel_bbox") or [None, None, None, None]
                if all(v is not None for v in pc):
                    pixel_center = (float(pc[0]), float(pc[1]))
                if all(v is not None for v in pb):
                    pixel_bbox = (int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]))
                placement_conf = float(cg.get("placement_confidence") or 0.0)
                coords_note = (
                    f"pixel_center={pixel_center} conf={placement_conf:.2f}"
                )
            except (TypeError, ValueError) as exc:
                per_findings.append(Finding(
                    issue_code="MALFORMED_PLACEMENT",
                    severity="warn",
                    object_id=aid,
                    message=f"could not parse coord-gen placement: {exc}",
                ))

            # Out-of-bounds check (pixel coords must lie in [0, IMG_SIZE_PX))
            if pixel_center is not None:
                cx, cy = pixel_center
                if not (0 <= cx < IMG_SIZE_PX and 0 <= cy < IMG_SIZE_PX):
                    per_findings.append(Finding(
                        issue_code="COORDS_OUT_OF_BOUNDS",
                        severity="warn",
                        object_id=aid,
                        field="pixel_center",
                        message=f"pixel_center {pixel_center} outside [0, {IMG_SIZE_PX})",
                        recommended_action="rerun coord-gen; treat placement as invalid",
                    ))
                    pixel_center = None  # invalidate so PLAN_CAPTURE skips
                    pixel_bbox = None

            if placement_conf is not None and placement_conf < 0.3:
                per_findings.append(Finding(
                    issue_code="LOW_CONFIDENCE",
                    severity="info",
                    object_id=aid,
                    field="placement_confidence",
                    message=f"placement_confidence={placement_conf:.2f} (<0.30)",
                ))

        # Derive lat/lon center if pixel placement is valid
        if pixel_center is not None:
            anchor_center_latlon = pixel_to_latlon(
                pixel_center[0], pixel_center[1],
                IMG_SIZE_PX, inputs.z16_center_latlon[0], inputs.z16_center_latlon[1],
                ZOOM,
            )
        else:
            anchor_center_latlon = inputs.z16_center_latlon  # fallback to cell center

        # Geometry from pixel_bbox; fall back to single-pixel rectangle on the
        # cell center if no valid bbox.
        geom_image = coords_meta["image_used"]
        if pixel_bbox is not None:
            geometry = _coord_bbox_to_geometry(
                pixel_bbox, inputs.z16_center_latlon, geom_image,
            )
        else:
            cx_fb = IMG_SIZE_PX // 2
            cy_fb = IMG_SIZE_PX // 2
            geometry = _coord_bbox_to_geometry(
                (cx_fb - 1, cy_fb - 1, cx_fb + 1, cy_fb + 1),
                inputs.z16_center_latlon, geom_image,
            )

        # PLAN_CAPTURE: build z18 fetch plan, but only for valid placements
        seed_plan: Z18FetchPlan | None = None
        plan_note = "skipped: no valid placement"
        if pixel_bbox is not None:
            extent_m = _bbox_extent_meters(pixel_bbox, anchor_center_latlon[0])
            seed_plan = z18_tile_plan_from_latlon(
                anchor_center_latlon, extent_m, tile_budget=config.tile_budget_z18,
            )
            plan_note = f"{len(seed_plan.tile_centers)} tiles, {seed_plan.extent_meters:.0f}m extent"

        # Phase history (3 events)
        phase_history = [
            discover_event,
            PhaseEvent(
                phase="C.COORDS", action="locate", actor=coords_actor,
                timestamp=coords_meta["ts"], note=coords_note,
            ),
            PhaseEvent(
                phase="C.PLAN_CAPTURE", action="plan",
                actor="mosaic.z18_tile_plan_from_latlon",
                timestamp=_utc_iso(), note=plan_note,
            ),
        ]

        confidence = float(v3a.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        scale = v3a.get("scale", "minor")
        if scale not in ("major", "minor"):
            scale = "minor"

        anchor_obj = AnchorStructure(
            anchor_id=aid,
            structure_type=v3a.get("structure_type", "other"),
            scale=scale,
            anchor_center_latlon=anchor_center_latlon,
            geometry=geometry,
            confidence=confidence,
            rationale=str(v3a.get("rationale", ""))[:1000],
            source_images_used=[
                discover_meta["z16_image_used"], coords_meta["image_used"],
            ],
            state="draft",
            phase_history=phase_history,
            provenance=_build_provenance(inputs, discover_meta, coords_meta),
            findings=per_findings,
            seed_z18_fetch_plan=seed_plan,
            priority_rank=rank,
            zone_id=v3a.get("zone_id"),
        )
        structures.append(anchor_obj)

    return structures, cell_findings


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def _persist(out_dir: Path, name: str, content: str | bytes) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _render_review_overlay(
    inputs: AnchorDiscoveryInputs,
    structures: list[AnchorStructure],
    output_dir: Path,
) -> Path | None:
    """Draw pipeline anchor bboxes (blue) + centers + labels on the z16 image.

    If a `gt_anchors.json` sits next to the bundle (i.e. one of the test cells),
    GT bboxes are drawn in red as well so the operator can eyeball pipeline-vs-GT
    coverage in one image. For novel cells (no GT) only the pipeline overlay
    is drawn — production callers don't need ground truth to use this.
    """
    from PIL import Image, ImageDraw  # local import to keep PIL out of the cold path

    from readwater.pipeline.structure.geo import latlon_to_pixel

    if not inputs.z16_image_path.exists():
        logger.warning("review overlay skipped: z16 image missing at %s",
                       inputs.z16_image_path)
        return None

    base = Image.open(inputs.z16_image_path).convert("RGBA")
    if base.size != (IMG_SIZE_PX, IMG_SIZE_PX):
        logger.warning("review overlay: z16 image dims %s != expected (1280,1280)",
                       base.size)

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _load_font(22)
    font_small = _load_font(16)

    GT_COLOR = (220, 30, 30, 255)
    PIPE_COLOR = (30, 110, 230, 255)

    # GT (optional, only when test fixture is present)
    gt_path = inputs.z16_grid_overlay_path.parent.parent / "structures" / inputs.cell_id / "gt_anchors.json"
    # Also try the bundle-conventional location used by anchor_discovery's own writes
    if not gt_path.exists():
        gt_path = output_dir / "gt_anchors.json"
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            for g in [a for a in gt.get("anchors", []) if a.get("status") == "active"]:
                bbox = tuple(g["pixel_bbox"])
                cx, cy = g["pixel_center"]
                draw.rectangle(bbox, outline=GT_COLOR, width=4)
                r = 9
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=GT_COLOR,
                             outline=(255, 255, 255, 255), width=2)
                tier = g.get("tier")
                label = f"{g['gt_id']} T{tier}" if tier is not None else g["gt_id"]
                tx, ty = bbox[0] + 4, max(bbox[1] - 26, 2)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            draw.text((tx + dx, ty + dy), label, fill="black", font=font)
                draw.text((tx, ty), label, fill=GT_COLOR, font=font)
        except Exception as exc:  # noqa: BLE001 — GT load failure shouldn't kill the overlay
            logger.warning("review overlay: GT load failed: %s", exc)

    # Pipeline anchors (always)
    for a in structures:
        # bbox from geometry.pixel_polygon (built from coord-gen pixel_bbox)
        if a.geometry and a.geometry.pixel_polygon:
            xs = [p[0] for p in a.geometry.pixel_polygon]
            ys = [p[1] for p in a.geometry.pixel_polygon]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                draw.rectangle(bbox, outline=PIPE_COLOR, width=3)
        # center: convert anchor_center_latlon back to pixel space
        cx, cy = latlon_to_pixel(
            a.anchor_center_latlon[0], a.anchor_center_latlon[1],
            IMG_SIZE_PX,
            inputs.z16_center_latlon[0], inputs.z16_center_latlon[1],
            ZOOM,
        )
        if 0 <= cx < IMG_SIZE_PX and 0 <= cy < IMG_SIZE_PX:
            r = 8
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PIPE_COLOR,
                         outline=(255, 255, 255, 255), width=2)
            stype = (a.structure_type or "?")[:14]
            label = f"{a.anchor_id} {stype}"
            tx, ty = int(cx) + 12, int(cy) - 8
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, fill="black", font=font_small)
            draw.text((tx, ty), label, fill=PIPE_COLOR, font=font_small)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{inputs.cell_id}_review_overlay.png"
    Image.alpha_composite(base, layer).convert("RGB").save(out_path)
    return out_path


def _pick_winner(
    nogrid_anchors: list[dict],
    grid_anchors: list[dict],
    winner: WinnerMode,
) -> tuple[list[dict], str]:
    if winner == "nogrid":
        return nogrid_anchors, "nogrid"
    return grid_anchors, "grid"


async def run_anchor_discovery(
    inputs: AnchorDiscoveryInputs,
    config: AnchorDiscoveryConfig,
    output_dir: Path | None = None,
) -> StructurePhaseResult:
    """Top-level entry. Runs DISCOVER → COORDS → PLAN_CAPTURE → ASSEMBLE.

    `output_dir`, if provided, receives raw responses + parsed JSON for both
    stages (and both variants if comparison mode). Otherwise nothing is
    persisted; the agent decides what to write.
    """
    cell_findings: list[Finding] = []

    # ---------- DISCOVER ----------
    if config.v3_mode == "comparison":
        v3_nogrid_parsed, v3_nogrid_raw, v3_nogrid_meta = await discover_anchors_v3(
            inputs, "nogrid", config.inject_evidence,
        )
        v3_grid_parsed, v3_grid_raw, v3_grid_meta = await discover_anchors_v3(
            inputs, "grid", config.inject_evidence,
        )
        if output_dir:
            _persist(output_dir, "v3_nogrid_raw.md", v3_nogrid_raw)
            _persist(output_dir, "v3_nogrid_parsed.json",
                     json.dumps(v3_nogrid_parsed, indent=2))
            _persist(output_dir, "v3_grid_raw.md", v3_grid_raw)
            _persist(output_dir, "v3_grid_parsed.json",
                     json.dumps(v3_grid_parsed, indent=2))
        chosen_v3, chosen_label = _pick_winner(
            v3_nogrid_parsed.get("anchors", []) or [],
            v3_grid_parsed.get("anchors", []) or [],
            config.v3_comparison_winner,
        )
        v3_anchors = chosen_v3
        discover_meta = v3_nogrid_meta if chosen_label == "nogrid" else v3_grid_meta
        cell_findings.append(Finding(
            issue_code="V3_COMPARISON_WINNER",
            severity="info",
            object_id=inputs.cell_id,
            message=(f"v3 comparison: nogrid={len(v3_nogrid_parsed.get('anchors',[]) or [])} "
                     f"vs grid={len(v3_grid_parsed.get('anchors',[]) or [])}; "
                     f"using '{chosen_label}' per config"),
        ))
    else:
        parsed, raw, discover_meta = await discover_anchors_v3(
            inputs, config.v3_mode, config.inject_evidence,
        )
        if output_dir:
            _persist(output_dir, f"v3_{config.v3_mode}_raw.md", raw)
            _persist(output_dir, f"v3_{config.v3_mode}_parsed.json",
                     json.dumps(parsed, indent=2))
        v3_anchors = parsed.get("anchors", []) or []

    if not v3_anchors:
        cell_findings.append(Finding(
            issue_code="V3_NO_ANCHORS",
            severity="info",
            object_id=inputs.cell_id,
            message="v3 returned zero anchors for this cell.",
        ))
        return StructurePhaseResult(
            cell_id=inputs.cell_id,
            anchors=[],
        )

    # ---------- COORDS ----------
    if config.coords_mode == "comparison":
        cg_nogrid_parsed, cg_nogrid_raw, cg_nogrid_meta = await locate_anchor_coords(
            inputs, v3_anchors, "nogrid",
        )
        cg_grid_parsed, cg_grid_raw, cg_grid_meta = await locate_anchor_coords(
            inputs, v3_anchors, "grid",
        )
        if output_dir:
            _persist(output_dir, "coords_nogrid_raw.md", cg_nogrid_raw)
            _persist(output_dir, "coords_nogrid_parsed.json",
                     json.dumps(cg_nogrid_parsed, indent=2))
            _persist(output_dir, "coords_grid_raw.md", cg_grid_raw)
            _persist(output_dir, "coords_grid_parsed.json",
                     json.dumps(cg_grid_parsed, indent=2))
        chosen_cg, chosen_label = _pick_winner(
            cg_nogrid_parsed.get("anchors", []) or [],
            cg_grid_parsed.get("anchors", []) or [],
            config.coords_comparison_winner,
        )
        coord_anchors = chosen_cg
        coords_meta = cg_nogrid_meta if chosen_label == "nogrid" else cg_grid_meta
        cell_findings.append(Finding(
            issue_code="COORDS_COMPARISON_WINNER",
            severity="info",
            object_id=inputs.cell_id,
            message=f"coords comparison run; using '{chosen_label}' per config",
        ))
    else:
        parsed_cg, raw_cg, coords_meta = await locate_anchor_coords(
            inputs, v3_anchors, config.coords_mode,
        )
        if output_dir:
            _persist(output_dir, f"coords_{config.coords_mode}_raw.md", raw_cg)
            _persist(output_dir, f"coords_{config.coords_mode}_parsed.json",
                     json.dumps(parsed_cg, indent=2))
        coord_anchors = parsed_cg.get("anchors", []) or []

    # ---------- PLAN_CAPTURE + ASSEMBLE ----------
    structures, assembly_findings = assemble_anchor_structures(
        inputs, config, v3_anchors, coord_anchors, discover_meta, coords_meta,
    )
    cell_findings.extend(assembly_findings)

    # ---------- REVIEW OVERLAY ----------
    if config.render_review_overlay and output_dir is not None:
        try:
            overlay_path = _render_review_overlay(inputs, structures, output_dir)
            if overlay_path:
                logger.info(
                    "[%s] review overlay: %s",
                    inputs.cell_id, overlay_path,
                )
        except Exception as exc:  # noqa: BLE001 — overlay is auxiliary
            logger.warning("[%s] review overlay render failed: %s", inputs.cell_id, exc)

    return StructurePhaseResult(
        cell_id=inputs.cell_id,
        anchors=structures,
    )


# ----------------------------------------------------------------------
# Synchronous wrapper for integration into agent.py
# ----------------------------------------------------------------------


def run_anchor_discovery_sync(
    inputs: AnchorDiscoveryInputs,
    config: AnchorDiscoveryConfig,
    output_dir: Path | None = None,
) -> StructurePhaseResult:
    """Sync wrapper for code paths that aren't async (legacy agent.py)."""
    return asyncio.run(run_anchor_discovery(inputs, config, output_dir))
