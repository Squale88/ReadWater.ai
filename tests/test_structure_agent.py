"""End-to-end tests for the Phase 1.5 grid-cell structure agent."""

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


@pytest.fixture
def z15_z16_images(tmp_path: Path):
    z15 = tmp_path / "z15.png"
    z16 = tmp_path / "z16.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(z15)
    Image.new("RGB", (64, 64), (30, 20, 10)).save(z16)
    return str(z15), str(z16)


# --- Canned responses ---


def _discover_response(num_anchors: int = 2, with_overlap: bool = False) -> dict:
    anchors = []
    for i in range(num_anchors):
        cells = [
            f"{chr(ord('C') + i)}3",
            f"{chr(ord('C') + i)}4",
            f"{chr(ord('D') + i)}3",
            f"{chr(ord('D') + i)}4",
        ]
        anchors.append({
            "anchor_id": f"a{i+1}",
            "structure_type": "drain" if i == 0 else "oyster_bar",
            "scale": "major" if i == 0 else "minor",
            "rationale": f"test anchor {i+1}",
            "cells": cells,
            "confidence": 0.9 - i * 0.1,
            "expected_relevance": 0.8 - i * 0.1,
            "truncated_by_edge": False,
            "continuation_edges": {
                "north": False, "south": False, "east": False, "west": False,
            },
            "expansion_priority": 0.1,
            "parent_context_needed": False,
        })
    if with_overlap and len(anchors) >= 2:
        # Same cells + same type -> same extraction result -> high IoU
        anchors[1]["cells"] = list(anchors[0]["cells"])
        anchors[1]["structure_type"] = anchors[0]["structure_type"]
    return {
        "summary": "test tile",
        "anchors": anchors,
        "_grid_rows": 8,
        "_grid_cols": 8,
        "_gridded_image_path": "/tmp/fake",
    }


def _resolve_resolved() -> dict:
    return {
        "extends": {"north": False, "south": False, "east": False, "west": False},
        "reason": "fully visible",
        "structure_resolved": True,
    }


def _identify_anchor_response(use_hull_escape: bool = False) -> dict:
    members = [
        {
            "name": "left flanking point",
            "feature_type": "point",
            "cells": ["C3", "D3"],
            "notes": "",
        },
        {
            "name": "receiving basin",
            "feature_type": "basin",
            "cells": ["F5", "F6", "G5"],
            "notes": "",
        },
    ]
    if use_hull_escape:
        influence = {
            "cells": "hull_of_anchor",
            "shape_type": "radial",
            "dominance_strength": 0.7,
            "bounded_by": [],
            "competing_structures": [],
        }
    else:
        influence = {
            "cells": ["C3", "C4", "D3", "D4", "D5", "E4", "E5", "F5", "F6"],
            "shape_type": "fan",
            "dominance_strength": 0.8,
            "bounded_by": ["shoreline"],
            "competing_structures": [],
        }
    return {
        "anchor": {
            "cells": ["D4", "D5", "E4", "E5"],
            "notes": "Y-junction",
        },
        "local_complex": {
            "members": members,
            "relationship_summary": "flanking features around the confluence",
        },
        "influence_zone": influence,
        "_grid_rows": 8,
        "_grid_cols": 8,
        "_gridded_image_path": "/tmp/fake",
    }


def _identify_subzones_response(whitelist_ok: bool = True) -> dict:
    subs = [
        {
            "subzone_id": "a1-s1",
            "subzone_type": "drain_throat",
            "cells": ["D4", "E4"],
            "relative_priority": 1.0,
            "reasoning_summary": "tightest funnel",
            "confidence": 0.85,
        },
        {
            "subzone_id": "a1-s2",
            "subzone_type": "point_tip",
            "cells": ["C3"],
            "relative_priority": 0.7,
            "reasoning_summary": "left flank ambush",
            "confidence": 0.75,
        },
    ]
    if not whitelist_ok:
        subs.append({
            "subzone_id": "a1-s3",
            "subzone_type": "receiving_basin_lane",  # NOT in whitelist
            "cells": ["F5", "F6"],
            "relative_priority": 0.5,
            "reasoning_summary": "",
            "confidence": 0.6,
        })
    return {
        "subzones": subs,
        "_grid_rows": 8,
        "_grid_cols": 8,
        "_gridded_image_path": "/tmp/fake",
    }


def _mocks(discover, resolve, identify_anchor_responses, identify_subzones_responses):
    if not isinstance(identify_anchor_responses, list):
        identify_anchor_responses = [identify_anchor_responses]
    if not isinstance(identify_subzones_responses, list):
        identify_subzones_responses = [identify_subzones_responses]

    def _side_effect_factory(responses):
        idx = {"i": 0}
        async def _fn(*args, **kwargs):
            r = responses[min(idx["i"], len(responses) - 1)]
            idx["i"] += 1
            return {**r, "raw_response": "mocked"}
        return _fn

    return {
        "discover_anchors": AsyncMock(return_value={**discover, "raw_response": "d"}),
        "resolve_continuation": AsyncMock(return_value={**resolve, "raw_response": "r"}),
        "identify_anchor": AsyncMock(side_effect=_side_effect_factory(identify_anchor_responses)),
        "identify_subzones": AsyncMock(side_effect=_side_effect_factory(identify_subzones_responses)),
    }


def _patch(mocks):
    return patch.multiple(
        "readwater.pipeline.structure.agent.llm",
        discover_anchors=mocks["discover_anchors"],
        resolve_continuation=mocks["resolve_continuation"],
        identify_anchor=mocks["identify_anchor"],
        identify_subzones=mocks["identify_subzones"],
    )


# --- Tests ---


async def test_end_to_end_happy_path(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="root-14-3",
            cell_center=CELL_CENTER,
            z15_image_path=z15,
            z16_image_path=z16,
            provider=provider,
            base_output_dir=tmp_path,
            parent_context="test parent",
        )

    assert len(result.anchors) == 1
    assert len(result.complexes) == 1
    assert len(result.influences) == 1
    assert len(result.subzones) == 2
    assert result.truncated is False

    anchor = result.anchors[0]
    assert anchor.geometry.extractor == "gridcell"
    assert anchor.geometry.extraction_mode == "corridor"  # drain → corridor
    assert anchor.geometry.seed_cells == ["D4", "D5", "E4", "E5"]
    assert anchor.geometry.grid_rows == 8
    assert anchor.geometry.grid_cols == 8

    influence = result.influences[0]
    assert influence.geometry.source == "llm_polygon"

    complex_ = result.complexes[0]
    assert len(complex_.members) == 2
    assert complex_.envelope is None
    for member in complex_.members:
        assert member.geometry.extractor == "gridcell"
        assert member.geometry.seed_cells

    for sz in result.subzones:
        assert sz.geometry.extractor == "gridcell"

    reg = json.loads(Path(result.registry_path).read_text())
    assert reg["processed"] == ["a1"]
    assert reg["final_accepted"]["anchors"] == ["a1"]
    annotated = Image.open(result.annotated_image_paths["a1"])
    assert annotated.size[0] >= 1280 and annotated.size[1] >= 1280


async def test_no_anchors_finalizes_immediately(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        {"summary": "", "anchors": [], "_grid_rows": 8, "_grid_cols": 8},
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="empty-cell",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    assert result.anchors == []
    assert mocks["identify_anchor"].await_count == 0
    assert mocks["identify_subzones"].await_count == 0
    assert Path(result.registry_path).exists()


async def test_deferred_anchors_persisted(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=5),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="deferred-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    assert len(result.deferred) == 2
    assert len(result.anchors) == 3
    reg = json.loads(Path(result.registry_path).read_text())
    assert len(reg["deferred"]) == 2


async def test_hull_of_anchor_escape_hatch(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        _identify_anchor_response(use_hull_escape=True),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="hull-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    assert len(result.influences) == 1
    assert result.influences[0].geometry.source == "convex_hull_of_anchor"


async def test_subzone_whitelist_enforced(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(whitelist_ok=False),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="whitelist-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    kept_ids = [sz.subzone_id for sz in result.subzones]
    assert "a1-s1" in kept_ids
    assert "a1-s2" in kept_ids
    assert "a1-s3" not in kept_ids

    reg = json.loads(Path(result.registry_path).read_text())
    rejection_ids = [f["feature_id"] for f in reg["failed_identifications"]]
    assert "a1-s3" in rejection_ids


async def test_cell_validation_regenerates_once(tmp_path, z15_z16_images):
    """First identify response has invalid cell labels; retry returns valid ones."""
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)

    bad = _identify_anchor_response()
    bad["anchor"] = {"cells": ["junk_label"], "notes": "bad"}
    good = _identify_anchor_response()

    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        [bad, good],
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="regen-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    assert mocks["identify_anchor"].await_count == 2
    assert len(result.anchors) == 1


async def test_overlap_audit_subordinates_heavily_overlapping_anchors(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=2, with_overlap=True),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="overlap-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    reg = json.loads(Path(result.registry_path).read_text())
    anchor_overlaps = [o for o in reg["overlap_report"] if o["level"] == "anchor"]
    assert anchor_overlaps
    assert any(o["iou"] > 0.5 for o in anchor_overlaps)
    assert result.subordinated_ids


async def test_budget_truncation(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    budget = StructureBudget(calls_per_anchor=1, tiles_per_anchor=50)
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="budget-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
            budget=budget,
        )
    # Only 1 call allowed per anchor; resolve_continuation eats it, so identify
    # is skipped and the anchor is truncated.
    assert result.truncated is True
    assert "a1" in result.truncated_ids


async def test_seed_cells_stored_on_geometry(tmp_path, z15_z16_images):
    z15, z16 = z15_z16_images
    provider = PlaceholderProvider(size=32)
    mocks = _mocks(
        _discover_response(num_anchors=1),
        _resolve_resolved(),
        _identify_anchor_response(),
        _identify_subzones_response(),
    )
    with _patch(mocks):
        result = await run_structure_phase(
            cell_id="cells-stored-test",
            cell_center=CELL_CENTER,
            z15_image_path=z15, z16_image_path=z16,
            provider=provider, base_output_dir=tmp_path,
        )
    anchor = result.anchors[0]
    assert anchor.geometry.seed_cells == ["D4", "D5", "E4", "E5"]

    for m in result.complexes[0].members:
        assert m.geometry.seed_cells
    for sz in result.subzones:
        assert sz.geometry.seed_cells
