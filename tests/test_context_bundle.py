"""Tests for visual-context utilities (Step 3).

All offline — uses synthetic PIL images and pure geo math. No LLM calls,
no provider fetches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from readwater.models.cell import BoundingBox
from readwater.pipeline.context_bundle import (
    _bbox_to_pixel_rect,
    draw_footprint_overlay,
)
from readwater.pipeline.structure.geo import (
    deg_lat_per_pixel,
    deg_lon_per_pixel,
)

BASE_CENTER = (26.0, -81.75)
BASE_ZOOM = 14
IMG_SIDE = 1280


def _make_base_image(tmp_path: Path, name: str = "base.png") -> Path:
    p = tmp_path / name
    img = Image.new("RGB", (IMG_SIDE, IMG_SIDE), color=(50, 50, 50))
    img.save(p)
    return p


def _centered_inner_bbox(width_px: int) -> BoundingBox:
    """Build a bbox whose projection onto the base image is a width_px-wide
    square centered on BASE_CENTER."""
    dlat = deg_lat_per_pixel(BASE_ZOOM, BASE_CENTER[0]) * width_px
    dlon = deg_lon_per_pixel(BASE_ZOOM, BASE_CENTER[0]) * width_px
    return BoundingBox(
        north=BASE_CENTER[0] + dlat / 2,
        south=BASE_CENTER[0] - dlat / 2,
        east=BASE_CENTER[1] + dlon / 2,
        west=BASE_CENTER[1] - dlon / 2,
    )


# --- _bbox_to_pixel_rect ---


def test_bbox_to_pixel_rect_centered_inner_maps_to_center_square():
    """A 320px-wide inner bbox centered on the base center should project to
    a [480, 480, 800, 800] square (+/- 1 px)."""
    inner = _centered_inner_bbox(width_px=320)
    x0, y0, x1, y1 = _bbox_to_pixel_rect(
        inner, BASE_CENTER, BASE_ZOOM, (IMG_SIDE, IMG_SIDE),
    )
    assert 479 <= x0 <= 481
    assert 479 <= y0 <= 481
    assert 799 <= x1 <= 801
    assert 799 <= y1 <= 801


def test_bbox_to_pixel_rect_northward_offset_shifts_y_up():
    """Shifting the inner bbox 200 px north should lower y0 and y1 by ~200."""
    dlat_200px = deg_lat_per_pixel(BASE_ZOOM, BASE_CENTER[0]) * 200
    base = _centered_inner_bbox(width_px=320)
    inner = BoundingBox(
        north=base.north + dlat_200px,
        south=base.south + dlat_200px,
        east=base.east,
        west=base.west,
    )
    x0, y0, x1, y1 = _bbox_to_pixel_rect(
        inner, BASE_CENTER, BASE_ZOOM, (IMG_SIDE, IMG_SIDE),
    )
    assert 279 <= y0 <= 281
    assert 599 <= y1 <= 601
    # x unchanged
    assert 479 <= x0 <= 481
    assert 799 <= x1 <= 801


def test_bbox_to_pixel_rect_clamps_when_outside():
    """A bbox fully outside the image clamps to the image edge without raising."""
    inner = BoundingBox(
        north=BASE_CENTER[0] + 10.0, south=BASE_CENTER[0] + 9.0,
        east=BASE_CENTER[1] + 10.0, west=BASE_CENTER[1] + 9.0,
    )
    x0, y0, x1, y1 = _bbox_to_pixel_rect(
        inner, BASE_CENTER, BASE_ZOOM, (IMG_SIDE, IMG_SIDE),
    )
    assert 0 <= x0 <= IMG_SIDE - 1
    assert 0 <= x1 <= IMG_SIDE - 1
    assert 0 <= y0 <= IMG_SIDE - 1
    assert 0 <= y1 <= IMG_SIDE - 1
    assert x0 <= x1 and y0 <= y1


# --- draw_footprint_overlay ---


def test_draw_footprint_overlay_writes_output(tmp_path):
    base = _make_base_image(tmp_path)
    inner = _centered_inner_bbox(width_px=320)
    out = tmp_path / "overlay.png"
    returned = draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(out), label="z16",
    )
    assert Path(returned).exists()
    assert Path(returned) == out


def test_draw_footprint_overlay_does_not_mutate_base(tmp_path):
    base = _make_base_image(tmp_path)
    before = hashlib.sha256(base.read_bytes()).hexdigest()
    inner = _centered_inner_bbox(width_px=320)
    out = tmp_path / "overlay.png"
    draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(out), label="z16",
    )
    after = hashlib.sha256(base.read_bytes()).hexdigest()
    assert before == after


def test_draw_footprint_overlay_draws_yellow_stroke(tmp_path):
    base = _make_base_image(tmp_path)
    inner = _centered_inner_bbox(width_px=320)
    out = tmp_path / "overlay.png"
    draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(out), label="",
    )
    overlay = Image.open(out).convert("RGB")
    pixels = list(overlay.getdata())
    yellow_count = sum(1 for r, g, b in pixels if r > 200 and g > 200 and b < 100)
    # Stroke width 6 on a ~320x320 rectangle -> ~8000 yellow pixels expected.
    # Use a conservative floor to avoid flakiness.
    assert yellow_count > 1000


def test_draw_footprint_overlay_creates_missing_parent_dir(tmp_path):
    base = _make_base_image(tmp_path)
    inner = _centered_inner_bbox(width_px=320)
    nested = tmp_path / "deep" / "nested" / "dir" / "overlay.png"
    path = draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(nested), label="",
    )
    assert Path(path).exists()


def test_draw_footprint_overlay_handles_fully_outside_bbox(tmp_path):
    base = _make_base_image(tmp_path)
    inner = BoundingBox(
        north=BASE_CENTER[0] + 10.0, south=BASE_CENTER[0] + 9.0,
        east=BASE_CENTER[1] + 10.0, west=BASE_CENTER[1] + 9.0,
    )
    out = tmp_path / "overlay.png"
    path = draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(out), label="",
    )
    assert Path(path).exists()


def test_draw_footprint_overlay_label_renders_near_rect(tmp_path):
    """With a label, the overlay must contain a black-backed badge somewhere."""
    base = _make_base_image(tmp_path)
    inner = _centered_inner_bbox(width_px=320)
    out = tmp_path / "overlay.png"
    draw_footprint_overlay(
        str(base), BASE_CENTER, BASE_ZOOM, inner, str(out), label="z16_self",
    )
    overlay = Image.open(out).convert("RGB")
    pixels = list(overlay.getdata())
    black_count = sum(1 for r, g, b in pixels if r < 10 and g < 10 and b < 10)
    # Base fill is (50, 50, 50) — never true-black — so any black pixels must
    # come from the label background.
    assert black_count > 50


# ---------------------------------------------------------------------------
# Step 6 — digests and build_cell_context
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, patch  # noqa: E402

from readwater.models.context import (  # noqa: E402
    CandidateFeatureThread,
    CellContext,
    DirectObservation,
    LineageRef,
    MorphologyInference,
)
from readwater.pipeline.context_bundle import (  # noqa: E402
    _collect_open_threads,
    _digest_ancestors,
    _digest_grid_scoring,
    _format_open_threads,
    build_cell_context,
)


# --- _digest_grid_scoring ---


def test_digest_grid_scoring_empty_returns_none_marker():
    assert _digest_grid_scoring(None) == "(none)"
    assert _digest_grid_scoring({}) == "(none)"


def test_digest_grid_scoring_counts_kept_ambiguous_pruned():
    scores = [{"cell_number": i, "score": s} for i, s in enumerate(
        [5, 5, 5, 5, 5, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], start=1,
    )]
    digest = _digest_grid_scoring({
        "summary": "Mixed bay",
        "hydrology_notes": "SE-flowing",
        "sub_scores": scores,
    })
    assert "Mixed bay" in digest
    assert "SE-flowing" in digest
    assert "kept=6" in digest
    assert "ambiguous=2" in digest
    assert "pruned=8" in digest


def test_digest_grid_scoring_skips_empty_fields():
    digest = _digest_grid_scoring({"summary": "", "hydrology_notes": "", "sub_scores": []})
    assert digest == "(none)"


# --- _digest_ancestors ---


def _lineage(cell_id: str, zoom: int, depth: int) -> LineageRef:
    return LineageRef(
        cell_id=cell_id, zoom=zoom, depth=depth,
        center=(26.0, -81.75),
        bbox=BoundingBox(north=27, south=26, east=-81, west=-82),
        image_path=f"{cell_id}.png",
    )


def test_digest_ancestors_empty_returns_none_marker():
    assert _digest_ancestors([], {}) == "(none)"


def test_digest_ancestors_renders_header_and_top_items():
    lineage = [_lineage("root", 12, 0), _lineage("root-6", 14, 1)]
    contexts = {
        "root": CellContext(
            cell_id="root", zoom=12,
            observations=[
                DirectObservation(observation_id="root:obs:0", label="mangrove_shoreline", location_hint="N"),
                DirectObservation(observation_id="root:obs:1", label="open_water"),
                DirectObservation(observation_id="root:obs:2", label="island"),
                DirectObservation(observation_id="root:obs:3", label="beach"),  # should be clipped (top 3)
            ],
            morphology=[
                MorphologyInference(
                    inference_id="root:morph:0",
                    kind="enclosed_by",
                    statement="Bay enclosed by barrier island.",
                ),
            ],
        ),
        "root-6": CellContext(cell_id="root-6", zoom=14),
    }
    block = _digest_ancestors(lineage, contexts)
    assert "z12 root" in block
    assert "z14 root-6" in block
    assert "mangrove_shoreline" in block
    assert "[N]" in block  # location hint rendered
    # Top-3 observations only — the 4th must be clipped.
    assert "beach" not in block
    assert "enclosed_by" in block


def test_digest_ancestors_marks_missing_context():
    lineage = [_lineage("root", 12, 0)]
    block = _digest_ancestors(lineage, {})
    assert "(no context)" in block


# --- _collect_open_threads + _format_open_threads ---


def test_collect_open_threads_filters_by_status_and_zoom():
    t_needs_z16 = CandidateFeatureThread(
        thread_id="root:th:0", feature_type="drain", status="hypothesized", needs_zoom=16,
    )
    t_needs_z18 = CandidateFeatureThread(
        thread_id="root:th:1", feature_type="point", status="hypothesized", needs_zoom=18,
    )
    t_resolved = CandidateFeatureThread(
        thread_id="root:th:2", feature_type="cove", status="resolved", needs_zoom=16,
    )
    t_no_zoom = CandidateFeatureThread(
        thread_id="root:th:3", feature_type="oyster_bar", status="hypothesized",
    )
    ctx = CellContext(
        cell_id="root", zoom=12,
        feature_threads=[t_needs_z16, t_needs_z18, t_resolved, t_no_zoom],
    )
    collected = _collect_open_threads({"root": ctx}, current_zoom=14)
    collected_ids = {t.thread_id for t in collected}
    # needs_zoom=16 >= 14 -> included; needs_zoom=18 >= 14 -> included
    # resolved -> excluded; no_zoom -> included
    assert collected_ids == {"root:th:0", "root:th:1", "root:th:3"}


def test_collect_open_threads_excludes_threads_already_past_zoom():
    t_past = CandidateFeatureThread(
        thread_id="root:th:0", feature_type="drain", status="hypothesized", needs_zoom=12,
    )
    ctx = CellContext(cell_id="root", zoom=12, feature_threads=[t_past])
    assert _collect_open_threads({"root": ctx}, current_zoom=14) == []


def test_format_open_threads_empty_returns_none_marker():
    assert _format_open_threads([]) == "(none)"


def test_format_open_threads_renders_id_type_and_summary():
    t = CandidateFeatureThread(
        thread_id="root:th:0", feature_type="drain",
        status="hypothesized", summary="south drain",
    )
    out = _format_open_threads([t])
    assert "root:th:0" in out
    assert "(drain)" in out
    assert "south drain" in out


# --- build_cell_context ---


def _mock_raw_payload():
    return {
        "observations": [
            {"label": "mangrove_shoreline", "location_hint": "S edge", "confidence": 0.8},
            {"label": "tidal_inlet", "location_hint": "SE", "confidence": 0.7},
            {"label": "", "location_hint": "ignored"},  # dropped — empty label
        ],
        "morphology": [
            {
                "kind": "drains_to",
                "statement": "Cuts drain to main channel.",
                "references": ["root:obs:1"],
                "confidence": 0.6,
            },
            {"kind": "", "statement": "bad"},  # dropped
        ],
        "feature_threads": [
            {
                "feature_type": "drain",
                "status": "hypothesized",
                "summary": "Possible drain.",
                "supporting_observations_local_idx": [1, 99],  # 99 is invalid
                "parent_thread_id": "root:th:9",  # dangling — must be dropped
                "needs_zoom": 18,
                "confidence": 0.55,
            },
            {"feature_type": "point", "status": ""},  # dropped
        ],
        "open_questions": [
            {"question": "Does the cut pinch at low tide?", "target_zoom": 18},
            {"question": ""},  # dropped
        ],
        "raw_response": "```json\n{\"observations\": []}\n```",
    }


async def test_build_cell_context_assigns_deterministic_ids(tmp_path):
    img = _make_base_image(tmp_path)
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value=_mock_raw_payload()),
    ):
        ctx = await build_cell_context(
            cell_id="root-6-11",
            zoom=16,
            image_path=str(img),
            center=BASE_CENTER,
            coverage_miles=0.37,
            ancestor_lineage=[],
            ancestor_contexts={},
            grid_scoring_result=None,
        )
    assert [o.observation_id for o in ctx.observations] == [
        "root-6-11:obs:0", "root-6-11:obs:1",
    ]
    assert [m.inference_id for m in ctx.morphology] == ["root-6-11:morph:0"]
    assert [t.thread_id for t in ctx.feature_threads] == ["root-6-11:th:0"]
    assert [q.question_id for q in ctx.open_questions] == ["root-6-11:q:0"]


async def test_build_cell_context_resolves_local_idx_to_observation_ids(tmp_path):
    img = _make_base_image(tmp_path)
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value=_mock_raw_payload()),
    ):
        ctx = await build_cell_context(
            cell_id="root-6-11", zoom=16, image_path=str(img),
            center=BASE_CENTER, coverage_miles=0.37,
        )
    thread = ctx.feature_threads[0]
    # local_idx=1 resolves to observations[1] = root-6-11:obs:1; local_idx=99 dropped
    assert thread.supporting_observation_ids == ["root-6-11:obs:1"]


async def test_build_cell_context_drops_dangling_parent_thread_id(tmp_path):
    img = _make_base_image(tmp_path)
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value=_mock_raw_payload()),
    ):
        ctx = await build_cell_context(
            cell_id="root-6-11", zoom=16, image_path=str(img),
            center=BASE_CENTER, coverage_miles=0.37,
        )
    assert ctx.feature_threads[0].parent_thread_id is None


async def test_build_cell_context_preserves_valid_parent_thread_id(tmp_path):
    img = _make_base_image(tmp_path)
    # Caller-provided open thread that matches the LLM's parent_thread_id.
    valid_parent = CandidateFeatureThread(
        thread_id="root:th:9",
        feature_type="drain",
        status="hypothesized",
        needs_zoom=18,
    )
    ancestor = CellContext(cell_id="root", zoom=12, feature_threads=[valid_parent])
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value=_mock_raw_payload()),
    ):
        ctx = await build_cell_context(
            cell_id="root-6-11", zoom=16, image_path=str(img),
            center=BASE_CENTER, coverage_miles=0.37,
            ancestor_lineage=[_lineage("root", 12, 0)],
            ancestor_contexts={"root": ancestor},
        )
    assert ctx.feature_threads[0].parent_thread_id == "root:th:9"


async def test_build_cell_context_writes_raw_markdown(tmp_path):
    img = _make_base_image(tmp_path, name="z0_6_11.png")
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value=_mock_raw_payload()),
    ):
        ctx = await build_cell_context(
            cell_id="root-6-11", zoom=16, image_path=str(img),
            center=BASE_CENTER, coverage_miles=0.37,
        )
    assert ctx.raw_response_path is not None
    raw_path = Path(ctx.raw_response_path)
    assert raw_path.exists()
    assert raw_path.name == "z0_6_11_context.md"
    assert "```json" in raw_path.read_text(encoding="utf-8")


async def test_build_cell_context_passes_digests_into_generator(tmp_path):
    """The generator must receive the grid-scoring digest and ancestor block."""
    img = _make_base_image(tmp_path)
    ancestor_ctx = CellContext(
        cell_id="root", zoom=12,
        observations=[DirectObservation(observation_id="root:obs:0", label="open_water")],
    )
    captured: dict = {}

    async def _fake_generate(**kwargs):
        captured.update(kwargs)
        return _mock_raw_payload()

    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=_fake_generate,
    ):
        await build_cell_context(
            cell_id="root-6-11", zoom=16, image_path=str(img),
            center=BASE_CENTER, coverage_miles=0.37,
            ancestor_lineage=[_lineage("root", 12, 0)],
            ancestor_contexts={"root": ancestor_ctx},
            grid_scoring_result={
                "summary": "Mixed bay",
                "sub_scores": [{"cell_number": i, "score": 5 if i < 4 else 0} for i in range(1, 17)],
            },
        )

    assert "Mixed bay" in captured["grid_scoring_digest"]
    assert "z12 root" in captured["ancestor_chain_block"]
    assert "open_water" in captured["ancestor_chain_block"]
    # No open threads fed in -> "(none)".
    assert captured["open_thread_block"] == "(none)"


async def test_build_cell_context_survives_empty_llm_response(tmp_path):
    img = _make_base_image(tmp_path)
    with patch(
        "readwater.pipeline.context_bundle.generate_cell_context",
        new=AsyncMock(return_value={"raw_response": ""}),
    ):
        ctx = await build_cell_context(
            cell_id="root", zoom=12, image_path=str(img),
            center=BASE_CENTER, coverage_miles=13.7,
        )
    assert ctx.observations == []
    assert ctx.morphology == []
    assert ctx.feature_threads == []
    assert ctx.open_questions == []
    assert ctx.raw_response_path is None


# ---------------------------------------------------------------------------
# Step 9 — Z16ContextBundle assembly
# ---------------------------------------------------------------------------


from readwater.models.context import VisualRole, Z16ContextBundle  # noqa: E402
from readwater.pipeline.context_bundle import (  # noqa: E402
    _role_from_zoom_delta,
    assemble_z16_bundle,
    bundle_path_for,
    persist_bundle,
)


def test_role_from_zoom_delta_exhaustive():
    # Pure delta lookup: delta=2 -> parent, delta=4 -> grandparent, delta=6 -> gg.
    assert _role_from_zoom_delta(16, 14) == VisualRole.Z14_PARENT
    assert _role_from_zoom_delta(16, 12) == VisualRole.Z12_GRANDPARENT
    assert _role_from_zoom_delta(16, 10) == VisualRole.Z10_GREAT_GRANDPARENT
    # Unsupported deltas return None.
    assert _role_from_zoom_delta(16, 8) is None
    assert _role_from_zoom_delta(16, 16) is None
    assert _role_from_zoom_delta(16, 15) is None


def _write_synthetic_base(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    img = Image.new("RGB", (IMG_SIDE, IMG_SIDE), color=(80, 80, 80))
    img.save(p)
    return p


def _synthetic_lineage_ref(
    cell_id: str, zoom: int, depth: int, image_path: Path,
) -> LineageRef:
    """LineageRef with a same-center bbox sized for the given zoom."""
    from readwater.pipeline.context_bundle import _same_center_bbox
    return LineageRef(
        cell_id=cell_id,
        zoom=zoom,
        depth=depth,
        center=BASE_CENTER,
        bbox=_same_center_bbox(BASE_CENTER, zoom),
        image_path=str(image_path),
        position_in_parent=None if depth == 0 else (0, 0),
    )


def test_assemble_bundle_z12_start_produces_4_visuals(tmp_path):
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_context_z15.png")

    ctx_root = CellContext(cell_id="root", zoom=12)
    ctx_z14 = CellContext(cell_id="root-6", zoom=14)
    ctx_z16 = CellContext(cell_id="root-6-11", zoom=16)

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=ctx_z16,
        ancestor_lineage=[z12, z14],
        ancestor_contexts={"root": ctx_root, "root-6": ctx_z14},
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )

    assert isinstance(bundle, Z16ContextBundle)
    assert set(bundle.visuals.keys()) == {
        VisualRole.Z16_LOCAL,
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
    }
    assert set(bundle.contexts.keys()) == {"root", "root-6", "root-6-11"}
    assert [r.cell_id for r in bundle.lineage] == ["root", "root-6", "root-6-11"]


def test_assemble_bundle_z10_start_produces_5_visuals(tmp_path):
    base_dir = tmp_path / "area"
    z10 = _synthetic_lineage_ref("root", 10, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z12 = _synthetic_lineage_ref("root-6", 12, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z14 = _synthetic_lineage_ref("root-6-11", 14, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z16 = _synthetic_lineage_ref("root-6-11-8", 16, 3, _write_synthetic_base(tmp_path, "z0_6_11_8.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_8_context_z15.png")

    contexts = {lr.cell_id: CellContext(cell_id=lr.cell_id, zoom=lr.zoom) for lr in (z10, z12, z14)}
    ctx_z16 = CellContext(cell_id="root-6-11-8", zoom=16)

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=ctx_z16,
        ancestor_lineage=[z10, z12, z14],
        ancestor_contexts=contexts,
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )
    assert set(bundle.visuals.keys()) == {
        VisualRole.Z16_LOCAL,
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
        VisualRole.Z10_GREAT_GRANDPARENT,
    }
    assert set(bundle.contexts.keys()) == {"root", "root-6", "root-6-11", "root-6-11-8"}


def test_assemble_bundle_overlay_files_are_written(tmp_path):
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_context_z15.png")

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=CellContext(cell_id="root-6-11", zoom=16),
        ancestor_lineage=[z12, z14],
        ancestor_contexts={
            "root": CellContext(cell_id="root", zoom=12),
            "root-6": CellContext(cell_id="root-6", zoom=14),
        },
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )

    # z16_local: no overlay.
    assert bundle.visuals[VisualRole.Z16_LOCAL].overlay_image_path is None
    # Other three visuals: overlay files on disk.
    for role in (
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
    ):
        ref = bundle.visuals[role]
        assert ref.overlay_image_path is not None
        assert Path(ref.overlay_image_path).exists()


def test_assemble_bundle_overlay_footprint_follows_next_inner_rule(tmp_path):
    """z14 draws z16; z12 draws z14; z15_same_center draws z16."""
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_context_z15.png")

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=CellContext(cell_id="root-6-11", zoom=16),
        ancestor_lineage=[z12, z14],
        ancestor_contexts={
            "root": CellContext(cell_id="root", zoom=12),
            "root-6": CellContext(cell_id="root-6", zoom=14),
        },
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )

    assert bundle.visuals[VisualRole.Z14_PARENT].overlay_footprint_bbox == z16.bbox
    assert bundle.visuals[VisualRole.Z12_GRANDPARENT].overlay_footprint_bbox == z14.bbox
    assert bundle.visuals[VisualRole.Z15_SAME_CENTER].overlay_footprint_bbox == z16.bbox


def test_assemble_bundle_omits_z15_when_none(tmp_path):
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=CellContext(cell_id="root-6-11", zoom=16),
        ancestor_lineage=[z12, z14],
        ancestor_contexts={
            "root": CellContext(cell_id="root", zoom=12),
            "root-6": CellContext(cell_id="root-6", zoom=14),
        },
        z15_same_center_path=None,
        base_output_dir=str(base_dir),
    )
    assert VisualRole.Z15_SAME_CENTER not in bundle.visuals
    # Other visuals still present.
    assert VisualRole.Z16_LOCAL in bundle.visuals
    assert VisualRole.Z14_PARENT in bundle.visuals
    assert VisualRole.Z12_GRANDPARENT in bundle.visuals


def test_persist_bundle_roundtrip(tmp_path):
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_context_z15.png")

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=CellContext(cell_id="root-6-11", zoom=16),
        ancestor_lineage=[z12, z14],
        ancestor_contexts={
            "root": CellContext(cell_id="root", zoom=12),
            "root-6": CellContext(cell_id="root-6", zoom=14),
        },
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )

    out_path = bundle_path_for(str(base_dir), "root-6-11")
    written = persist_bundle(bundle, out_path)
    assert Path(written).exists()
    restored = Z16ContextBundle.model_validate_json(Path(written).read_text(encoding="utf-8"))
    assert restored == bundle


def test_bundle_path_for_uses_canonical_layout(tmp_path):
    p = bundle_path_for(str(tmp_path), "root-6-11")
    assert p == tmp_path / "structures" / "root-6-11" / "context_bundle.json"


# ---------------------------------------------------------------------------
# Step 10 — load_bundle seam
# ---------------------------------------------------------------------------


from readwater.pipeline.context_bundle import load_bundle  # noqa: E402


def test_load_bundle_roundtrips_persisted_bundle(tmp_path):
    base_dir = tmp_path / "area"
    z12 = _synthetic_lineage_ref("root", 12, 0, _write_synthetic_base(tmp_path, "z0.png"))
    z14 = _synthetic_lineage_ref("root-6", 14, 1, _write_synthetic_base(tmp_path, "z0_6.png"))
    z16 = _synthetic_lineage_ref("root-6-11", 16, 2, _write_synthetic_base(tmp_path, "z0_6_11.png"))
    z15_same_center = _write_synthetic_base(tmp_path, "z0_6_11_context_z15.png")

    bundle = assemble_z16_bundle(
        self_lineage=z16,
        self_context=CellContext(cell_id="root-6-11", zoom=16),
        ancestor_lineage=[z12, z14],
        ancestor_contexts={
            "root": CellContext(cell_id="root", zoom=12),
            "root-6": CellContext(cell_id="root-6", zoom=14),
        },
        z15_same_center_path=str(z15_same_center),
        base_output_dir=str(base_dir),
    )
    out = bundle_path_for(str(base_dir), "root-6-11")
    persist_bundle(bundle, out)

    restored = load_bundle(out)
    assert restored == bundle


def test_structure_agent_is_byte_identical_to_master():
    """Phase-1 hard rule: no edits at all to structure/agent.py.

    Compares the local file to the master branch via git. This is a
    regression guard — if someone makes any change to agent.py in this
    branch, the test fails.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "diff", "master", "--", "src/readwater/pipeline/structure/agent.py"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"git diff failed: {result.stderr}"
    assert result.stdout == "", (
        "structure/agent.py has diverged from master in this branch, "
        "which is forbidden in Phase 1:\n" + result.stdout
    )
