"""End-to-end-ish tests for the structure-phase state machine.

Mocks the four Claude vision calls at the `readwater.pipeline.structure.prompts`
layer; uses PlaceholderProvider for actual image "fetches". Asserts:
  - state transitions (DISCOVER + per-anchor steps)
  - registry.json written with expected fields
  - annotated PNG produced per anchor
  - deferred anchors persisted, not dropped
  - overlap audit fires when anchors overlap
  - convex_hull_of_anchor escape hatch works
  - budget decremented correctly
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from readwater.api.providers.placeholder import PlaceholderProvider
from readwater.pipeline.structure.agent import (
    StructureBudget,
    run_structure_phase,
)

CELL_CENTER = (27.5, -82.0)


def _make_square_bbox(cx: int, cy: int, side: int) -> list[int]:
    return [cx - side // 2, cy - side // 2, side, side]


def _make_square_poly(cx: int, cy: int, side: int) -> list[list[int]]:
    half = side // 2
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


@pytest.fixture
def z15_z16_images(tmp_path: Path):
    """Two placeholder PNGs standing in for zoom-15 parent and zoom-16 cell."""
    z15 = tmp_path / "z15.png"
    z16 = tmp_path / "z16.png"
    # Small PNGs — Mosaic resizes tiles to 1280 anyway, and these two are only
    # passed to the (mocked) discover_anchors call.
    Image.new("RGB", (64, 64), (10, 20, 30)).save(z15)
    Image.new("RGB", (64, 64), (30, 20, 10)).save(z16)
    return str(z15), str(z16)


def _discover_response(num_anchors: int = 2, with_overlap: bool = False) -> dict:
    anchors = []
    for i in range(num_anchors):
        cx = 400 + i * 350
        bbox = _make_square_bbox(cx, 640, 200)
        anchors.append({
            "anchor_id": f"a{i+1}",
            "structure_type": "drain" if i == 0 else "point",
            "scale": "major" if i == 0 else "minor",
            "rationale": f"test anchor {i+1}",
            "approx_bbox_px_z16": bbox,
            "orientation_deg": 90,
            "confidence": 0.9 - i * 0.1,
            "expected_relevance": 0.8 - i * 0.1,
            "truncated_by_edge": False,
            "continuation_edges": {"north": False, "south": False, "east": False, "west": False},
            "expansion_priority": 0.1,
            "parent_context_needed": False,
        })
    if with_overlap and len(anchors) >= 2:
        # Make a2 heavily overlap a1 in z16 bbox space
        anchors[1]["approx_bbox_px_z16"] = list(anchors[0]["approx_bbox_px_z16"])
    return {"summary": "test tile", "anchors": anchors}


def _resolve_response_resolved() -> dict:
    return {
        "extends": {"north": False, "south": False, "east": False, "west": False},
        "reason": "fully visible",
        "structure_resolved": True,
    }


def _influence_response(use_hull_escape: bool = False) -> dict:
    # Assume mosaic is ~1280x1280 for a single-tile anchor; polygons are conservative.
    anchor_poly = _make_square_poly(640, 640, 400)
    complex_poly = _make_square_poly(640, 640, 700)
    if use_hull_escape:
        influence = {"convex_hull_of_anchor": True, "expand_px": 100}
    else:
        influence = _make_square_poly(640, 640, 900)
    return {
        "anchor_polygon_px": anchor_poly,
        "local_complex": {
            "member_features": ["point", "basin"],
            "polygon_px": complex_poly,
            "relationship_summary": "drain and basin",
        },
        "influence_zone": {
            "polygon_px": influence,
            "shape_type": "fan",
            "dominance_strength": 0.8,
            "bounded_by": ["shoreline"],
            "competing_structures": [],
        },
    }


def _subzones_response() -> dict:
    return {
        "subzones": [
            {
                "subzone_id": "a1-s1",
                "subzone_type": "drain_throat",
                "polygon_px": _make_square_poly(640, 640, 80),
                "relative_priority": 1.0,
                "reasoning_summary": "current funnel",
                "confidence": 0.85,
            },
            {
                "subzone_id": "a1-s2",
                "subzone_type": "left_ambush_point",
                "polygon_px": _make_square_poly(500, 640, 120),
                "relative_priority": 0.7,
                "reasoning_summary": "flanking point",
                "confidence": 0.75,
            },
        ]
    }


def _mock_llm_module(
    discover: dict,
    resolve: dict,
    influence: dict,
    subzones: dict,
):
    return {
        "discover_anchors": AsyncMock(return_value={**discover, "raw_response": "d"}),
        "resolve_continuation": AsyncMock(return_value={**resolve, "raw_response": "r"}),
        "model_influence": AsyncMock(return_value={**influence, "raw_response": "i"}),
        "define_subzones": AsyncMock(return_value={**subzones, "raw_response": "s"}),
    }


async def test_end_to_end_happy_path(tmp_path: Path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    mocks = _mock_llm_module(
        _discover_response(num_anchors=1),
        _resolve_response_resolved(),
        _influence_response(),
        _subzones_response(),
    )

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="root-14-3",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
            parent_context="test parent",
        )

    # --- Core result shape ---
    assert len(result.anchors) == 1
    assert len(result.complexes) == 1
    assert len(result.influences) == 1
    assert len(result.subzones) == 2
    assert result.truncated is False

    # --- Polygon lengths match between pixel and latlon ---
    for a in result.anchors:
        assert len(a.geometry.pixel_polygon) == len(a.geometry.latlon_polygon)
    for s in result.subzones:
        assert len(s.geometry.pixel_polygon) == len(s.geometry.latlon_polygon)

    # --- Registry written ---
    registry_path = Path(result.registry_path)
    assert registry_path.exists()
    reg = json.loads(registry_path.read_text())
    assert reg["cell_id"] == "root-14-3"
    assert reg["processed"] == ["a1"]
    assert reg["discovered"]
    assert reg["final_accepted"]["anchors"] == ["a1"]

    # --- Annotated PNG exists and has mosaic dimensions ---
    annotated = Path(result.annotated_image_paths["a1"])
    assert annotated.exists()
    img = Image.open(annotated)
    assert img.size[0] >= 1280 and img.size[1] >= 1280

    # --- LLM call counts: 1 discover + 1 resolve + 1 influence + 1 subzones
    assert mocks["discover_anchors"].await_count == 1
    assert mocks["resolve_continuation"].await_count == 1
    assert mocks["model_influence"].await_count == 1
    assert mocks["define_subzones"].await_count == 1
    assert result.api_calls_used == 4


async def test_deferred_anchors_persisted(tmp_path: Path, z15_z16_images):
    """Discovery returns 5 anchors; only top 3 processed, rest recorded as deferred."""
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    mocks = _mock_llm_module(
        _discover_response(num_anchors=5),
        _resolve_response_resolved(),
        _influence_response(),
        _subzones_response(),
    )

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="root-14-3",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
        )

    assert len(result.deferred) == 2
    assert len(result.anchors) == 3
    reg = json.loads(Path(result.registry_path).read_text())
    assert len(reg["deferred"]) == 2


async def test_no_anchors_finalizes_immediately(tmp_path: Path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    empty = {"summary": "featureless", "anchors": []}
    mocks = _mock_llm_module(empty, _resolve_response_resolved(), _influence_response(), _subzones_response())

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="empty-cell",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
        )

    assert result.anchors == []
    assert mocks["resolve_continuation"].await_count == 0
    assert mocks["model_influence"].await_count == 0
    assert Path(result.registry_path).exists()


async def test_convex_hull_escape_hatch(tmp_path: Path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    mocks = _mock_llm_module(
        _discover_response(num_anchors=1),
        _resolve_response_resolved(),
        _influence_response(use_hull_escape=True),
        _subzones_response(),
    )

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="hull-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
        )

    assert len(result.influences) == 1
    influence = result.influences[0]
    # The dilated hull of the anchor should still have at least 3 vertices
    assert len(influence.geometry.pixel_polygon) >= 3


async def test_overlap_audit_on_overlapping_anchors(tmp_path: Path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    mocks = _mock_llm_module(
        _discover_response(num_anchors=2, with_overlap=True),
        _resolve_response_resolved(),
        _influence_response(),
        _subzones_response(),
    )

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="overlap-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
        )

    reg = json.loads(Path(result.registry_path).read_text())
    assert reg["overlap_report"], "expected at least one overlap entry"
    # At least one anchor-level overlap should have been flagged
    anchor_overlaps = [o for o in reg["overlap_report"] if o["level"] == "anchor"]
    assert anchor_overlaps
    # IoU should be substantial since the two anchors share identical bboxes
    assert any(o["iou"] > 0.5 for o in anchor_overlaps)


async def test_normalized_coords_are_scaled(tmp_path: Path, z15_z16_images):
    """LLM returns polygons in [0,1] normalized space; agent scales to pixel frame."""
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    # Discovery with normalized bbox (approx_bbox_frac)
    discover = {
        "summary": "normalized test",
        "anchors": [{
            "anchor_id": "a1",
            "structure_type": "drain",
            "scale": "major",
            "rationale": "test",
            "approx_bbox_frac": [0.3, 0.3, 0.3, 0.3],
            "orientation_deg": 0,
            "confidence": 0.9,
            "expected_relevance": 0.9,
            "truncated_by_edge": False,
            "continuation_edges": {"north": False, "south": False, "east": False, "west": False},
            "expansion_priority": 0.1,
            "parent_context_needed": False,
        }],
    }
    # Model influence with normalized polygons
    influence = {
        "anchor_polygon": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        "local_complex": {
            "member_features": [],
            "polygon": [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]],
            "relationship_summary": "",
        },
        "influence_zone": {
            "polygon": [[0.30, 0.30], [0.70, 0.30], [0.70, 0.70], [0.30, 0.70]],
            "shape_type": "radial",
            "dominance_strength": 0.7,
            "bounded_by": [],
            "competing_structures": [],
        },
    }
    subzones = {
        "subzones": [{
            "subzone_id": "a1-s1",
            "subzone_type": "drain_throat",
            "polygon": [[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55]],
            "relative_priority": 1.0,
            "reasoning_summary": "",
            "confidence": 0.9,
        }],
    }

    mocks = _mock_llm_module(discover, _resolve_response_resolved(), influence, subzones)

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="norm-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
        )

    assert len(result.anchors) == 1
    anchor = result.anchors[0]
    # Polygon was specified as normalized [0.4-0.6]. After scaling to the
    # mosaic canvas, it should occupy ~40-60% on both axes. Get mosaic size
    # from the annotated image.
    mosaic_img = Image.open(result.mosaic_image_paths["a1"])
    mw, mh = mosaic_img.size
    xs = [p[0] for p in anchor.geometry.pixel_polygon]
    ys = [p[1] for p in anchor.geometry.pixel_polygon]
    assert min(xs) / mw == pytest.approx(0.4, abs=0.02)
    assert max(xs) / mw == pytest.approx(0.6, abs=0.02)
    assert min(ys) / mh == pytest.approx(0.4, abs=0.02)
    assert max(ys) / mh == pytest.approx(0.6, abs=0.02)


async def test_budget_caps_calls(tmp_path: Path, z15_z16_images):
    """Very low per-anchor budget should force early FINALIZE with truncated=True."""
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    mocks = _mock_llm_module(
        _discover_response(num_anchors=1),
        _resolve_response_resolved(),
        _influence_response(),
        _subzones_response(),
    )
    budget = StructureBudget(calls_per_anchor=1, tiles_per_anchor=50)

    with patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        model_influence=mocks["model_influence"],
        define_subzones=mocks["define_subzones"],
    ):
        result = await run_structure_phase(
            cell_id="budget-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
            budget=budget,
        )

    # With only 1 call allowed per anchor and the first resolve-continuation
    # eating that call, MODEL_INFLUENCE must be skipped. Anchor gets marked
    # truncated with no final geometry.
    assert result.truncated is True
    assert "a1" in result.truncated_ids
