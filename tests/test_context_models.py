"""Unit tests for retained-cell context models (Phase 1, Step 1).

Covers instantiation, required fields, confidence/coverage bounds, ID format
sanity, enum membership, keyed-mapping correctness on the bundle, JSON
round-trip, and the depicts_bbox vs overlay_footprint_bbox disambiguation.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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

SAMPLE_BBOX = BoundingBox(north=26.05, south=26.02, east=-81.72, west=-81.76)
SAMPLE_CENTER = (26.035, -81.74)


# --- VisualRole ---


def test_visual_role_membership():
    values = {r.value for r in VisualRole}
    assert values == {
        "z16_local",
        "z15_same_center",
        "z14_parent",
        "z12_grandparent",
        "z10_great_grandparent",
    }


def test_visual_role_is_string_enum():
    assert VisualRole.Z16_LOCAL == "z16_local"
    assert isinstance(VisualRole.Z14_PARENT.value, str)


# --- LineageRef ---


def test_lineageref_root_has_no_position_in_parent():
    ref = LineageRef(
        cell_id="root", zoom=12, depth=0,
        center=SAMPLE_CENTER, bbox=SAMPLE_BBOX,
        image_path="data/areas/test/images/z0.png",
    )
    assert ref.position_in_parent is None


def test_lineageref_child_records_position_in_parent():
    ref = LineageRef(
        cell_id="root-6", zoom=14, depth=1,
        center=SAMPLE_CENTER, bbox=SAMPLE_BBOX,
        image_path="data/areas/test/images/z0_6.png",
        position_in_parent=(1, 1),
    )
    assert ref.position_in_parent == (1, 1)


def test_lineageref_negative_depth_rejected():
    with pytest.raises(ValidationError):
        LineageRef(
            cell_id="x", zoom=12, depth=-1,
            center=SAMPLE_CENTER, bbox=SAMPLE_BBOX,
        )


# --- ID patterns accept deterministic format ---


def test_direct_observation_accepts_deterministic_id():
    obs = DirectObservation(
        observation_id="root-6-11:obs:0",
        label="mangrove_shoreline",
    )
    assert obs.observation_id == "root-6-11:obs:0"


def test_morphology_accepts_deterministic_id():
    m = MorphologyInference(
        inference_id="root-6-11:morph:0",
        kind="drains_to",
        statement="Drains to the main basin.",
    )
    assert m.inference_id == "root-6-11:morph:0"


def test_thread_accepts_deterministic_id_and_parent_link():
    t = CandidateFeatureThread(
        thread_id="root-6-11:th:0",
        feature_type="drain",
        status="hypothesized",
        parent_thread_id="root-6:th:2",
    )
    assert t.parent_thread_id == "root-6:th:2"


def test_question_accepts_deterministic_id():
    q = UnresolvedQuestion(
        question_id="root-6-11:q:0",
        question="Does this cut have a defined throat?",
    )
    assert q.question_id == "root-6-11:q:0"


# --- Bounds enforcement ---


def test_confidence_above_1_rejected_on_observation():
    with pytest.raises(ValidationError):
        DirectObservation(observation_id="x:obs:0", label="foo", confidence=1.5)


def test_confidence_below_0_rejected_on_thread():
    with pytest.raises(ValidationError):
        CandidateFeatureThread(
            thread_id="x:th:0",
            feature_type="drain",
            status="hypothesized",
            confidence=-0.1,
        )


def test_evidence_coverage_fraction_above_1_rejected():
    with pytest.raises(ValidationError):
        EvidenceSummary(layer="water", coverage_fraction=1.2)


def test_evidence_coverage_fraction_negative_rejected():
    with pytest.raises(ValidationError):
        EvidenceSummary(layer="water", coverage_fraction=-0.01)


# --- CellContext ---


def test_cellcontext_defaults_empty_lists():
    ctx = CellContext(cell_id="root-6-11", zoom=16)
    assert ctx.observations == []
    assert ctx.morphology == []
    assert ctx.feature_threads == []
    assert ctx.open_questions == []
    assert ctx.evidence == []
    assert ctx.model_used == ""
    assert ctx.source_images == []
    assert ctx.raw_response_path is None


def test_cellcontext_populated_roundtrip():
    obs = DirectObservation(
        observation_id="root-6-11:obs:0",
        label="mangrove_shoreline",
        location_hint="S edge",
        confidence=0.8,
    )
    morph = MorphologyInference(
        inference_id="root-6-11:morph:0",
        kind="drains_to",
        statement="The cut on the SE drains to the main channel visible at z14.",
        references=["root-6-11:obs:0", "root-6"],
        confidence=0.6,
    )
    thread = CandidateFeatureThread(
        thread_id="root-6-11:th:0",
        feature_type="drain",
        status="hypothesized",
        summary="Possible drain throat on SE corner.",
        supporting_observation_ids=["root-6-11:obs:0"],
        parent_thread_id="root-6:th:2",
        needs_zoom=18,
        confidence=0.55,
    )
    q = UnresolvedQuestion(
        question_id="root-6-11:q:0",
        question="Does this cut have a defined throat?",
        target_zoom=18,
    )
    ev = EvidenceSummary(layer="water", coverage_fraction=0.62, notes="NDWI")
    ctx = CellContext(
        cell_id="root-6-11",
        zoom=16,
        observations=[obs],
        morphology=[morph],
        feature_threads=[thread],
        open_questions=[q],
        evidence=[ev],
        model_used="claude-sonnet-4-20250514",
        source_images=["z0_6_11.png"],
    )
    dumped = ctx.model_dump_json()
    restored = CellContext.model_validate_json(dumped)
    assert restored == ctx


# --- VisualContextRef ---


def test_visualref_z16_local_has_no_overlay_by_default():
    ref = VisualContextRef(
        role=VisualRole.Z16_LOCAL,
        zoom=16,
        center=SAMPLE_CENTER,
        depicts_bbox=SAMPLE_BBOX,
        base_image_path="z0_6_11.png",
        lineage_cell_id="root-6-11",
    )
    assert ref.overlay_image_path is None
    assert ref.overlay_footprint_bbox is None
    assert ref.overlay_draws == []


def test_visualref_z15_same_center_has_none_lineage_cell_id():
    ref = VisualContextRef(
        role=VisualRole.Z15_SAME_CENTER,
        zoom=15,
        center=SAMPLE_CENTER,
        depicts_bbox=SAMPLE_BBOX,
        overlay_footprint_bbox=SAMPLE_BBOX,
        base_image_path="z0_6_11_context_z15.png",
        overlay_image_path="overlay_z15_same_center.png",
        overlay_draws=["z16_footprint"],
        lineage_cell_id=None,
    )
    assert ref.lineage_cell_id is None


def test_visualref_depicts_vs_overlay_footprint_are_distinct_fields():
    outer = BoundingBox(north=10, south=0, east=10, west=0)
    inner = BoundingBox(north=6, south=4, east=6, west=4)
    parent = VisualContextRef(
        role=VisualRole.Z14_PARENT,
        zoom=14,
        center=SAMPLE_CENTER,
        depicts_bbox=outer,
        overlay_footprint_bbox=inner,
        base_image_path="z0_6.png",
        overlay_image_path="overlay_z14_parent.png",
        overlay_draws=["z16_footprint"],
        lineage_cell_id="root-6",
    )
    assert parent.depicts_bbox == outer
    assert parent.overlay_footprint_bbox == inner
    assert parent.depicts_bbox != parent.overlay_footprint_bbox


# --- Z16ContextBundle ---


def _make_bundle(start_zoom: int) -> Z16ContextBundle:
    """Build a representative bundle for the given start zoom.

    z12-start -> lineage [z12_root, z14_parent, z16_self] (3),
                 visuals {z16_local, z15_same_center, z14_parent, z12_grandparent} (4),
                 contexts {root, root-6, root-6-11} (3)
    z10-start -> lineage [z10_root, z12, z14, z16_self] (4),
                 visuals {..., z10_great_grandparent} (5),
                 contexts {root, root-6, root-6-11, root-6-11-8} (4)
    """
    if start_zoom == 10:
        refs = [
            LineageRef(cell_id="root", zoom=10, depth=0, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0.png"),
            LineageRef(cell_id="root-6", zoom=12, depth=1, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0_6.png", position_in_parent=(1, 1)),
            LineageRef(cell_id="root-6-11", zoom=14, depth=2, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0_6_11.png", position_in_parent=(2, 2)),
            LineageRef(cell_id="root-6-11-8", zoom=16, depth=3, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0_6_11_8.png", position_in_parent=(1, 3)),
        ]
    else:
        refs = [
            LineageRef(cell_id="root", zoom=12, depth=0, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0.png"),
            LineageRef(cell_id="root-6", zoom=14, depth=1, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0_6.png", position_in_parent=(1, 1)),
            LineageRef(cell_id="root-6-11", zoom=16, depth=2, center=SAMPLE_CENTER, bbox=SAMPLE_BBOX, image_path="z0_6_11.png", position_in_parent=(2, 2)),
        ]

    contexts = {ref.cell_id: CellContext(cell_id=ref.cell_id, zoom=ref.zoom) for ref in refs}
    leaf = refs[-1]
    z14_ref = next(r for r in refs if r.zoom == 14)
    z12_ref = next(r for r in refs if r.zoom == 12)

    visuals: dict[VisualRole, VisualContextRef] = {
        VisualRole.Z16_LOCAL: VisualContextRef(
            role=VisualRole.Z16_LOCAL, zoom=16, center=SAMPLE_CENTER,
            depicts_bbox=leaf.bbox, base_image_path=leaf.image_path or "",
            lineage_cell_id=leaf.cell_id,
        ),
        VisualRole.Z15_SAME_CENTER: VisualContextRef(
            role=VisualRole.Z15_SAME_CENTER, zoom=15, center=SAMPLE_CENTER,
            depicts_bbox=leaf.bbox, overlay_footprint_bbox=leaf.bbox,
            base_image_path="z0_6_11_context_z15.png",
            overlay_image_path="overlay_z15_same_center.png",
            overlay_draws=["z16_footprint"],
            lineage_cell_id=None,
        ),
        VisualRole.Z14_PARENT: VisualContextRef(
            role=VisualRole.Z14_PARENT, zoom=14, center=SAMPLE_CENTER,
            depicts_bbox=z14_ref.bbox, overlay_footprint_bbox=leaf.bbox,
            base_image_path=z14_ref.image_path or "",
            overlay_image_path="overlay_z14_parent.png",
            overlay_draws=["z16_footprint"],
            lineage_cell_id=z14_ref.cell_id,
        ),
        VisualRole.Z12_GRANDPARENT: VisualContextRef(
            role=VisualRole.Z12_GRANDPARENT, zoom=12, center=SAMPLE_CENTER,
            depicts_bbox=z12_ref.bbox, overlay_footprint_bbox=z14_ref.bbox,
            base_image_path=z12_ref.image_path or "",
            overlay_image_path="overlay_z12_grandparent.png",
            overlay_draws=["z14_footprint"],
            lineage_cell_id=z12_ref.cell_id,
        ),
    }
    if start_zoom == 10:
        z10_ref = refs[0]
        visuals[VisualRole.Z10_GREAT_GRANDPARENT] = VisualContextRef(
            role=VisualRole.Z10_GREAT_GRANDPARENT, zoom=10, center=SAMPLE_CENTER,
            depicts_bbox=z10_ref.bbox, overlay_footprint_bbox=z12_ref.bbox,
            base_image_path=z10_ref.image_path or "",
            overlay_image_path="overlay_z10_great_grandparent.png",
            overlay_draws=["z12_footprint"],
            lineage_cell_id=z10_ref.cell_id,
        )

    return Z16ContextBundle(
        cell_id=leaf.cell_id,
        compiled_at="2026-04-20T12:00:00Z",
        lineage=refs,
        visuals=visuals,
        contexts=contexts,
    )


def test_bundle_z12_start_has_4_visuals_and_3_contexts():
    bundle = _make_bundle(start_zoom=12)
    assert set(bundle.visuals.keys()) == {
        VisualRole.Z16_LOCAL,
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
    }
    assert set(bundle.contexts.keys()) == {"root", "root-6", "root-6-11"}
    assert len(bundle.lineage) == 3


def test_bundle_z10_start_has_5_visuals_and_4_contexts():
    bundle = _make_bundle(start_zoom=10)
    assert set(bundle.visuals.keys()) == {
        VisualRole.Z16_LOCAL,
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
        VisualRole.Z10_GREAT_GRANDPARENT,
    }
    assert len(bundle.contexts) == 4
    assert len(bundle.lineage) == 4


def test_bundle_z15_same_center_has_no_context_entry():
    """z15_same_center is visual-only: no lineage entry, no context entry."""
    bundle = _make_bundle(start_zoom=12)
    z15 = bundle.visuals[VisualRole.Z15_SAME_CENTER]
    assert z15.lineage_cell_id is None
    assert all(ref.zoom != 15 for ref in bundle.lineage)
    # Every context key maps to a lineage cell_id
    lineage_ids = {ref.cell_id for ref in bundle.lineage}
    assert set(bundle.contexts.keys()) == lineage_ids


def test_bundle_visuals_keyed_by_role_not_by_position():
    bundle = _make_bundle(start_zoom=12)
    assert bundle.visuals[VisualRole.Z14_PARENT].zoom == 14
    assert bundle.visuals[VisualRole.Z12_GRANDPARENT].zoom == 12
    assert bundle.visuals[VisualRole.Z16_LOCAL].zoom == 16


def test_bundle_overlay_footprint_follows_next_inner_rule():
    """Each ancestor visual draws the NEXT-INNER retained region.
       z14 draws z16; z12 draws z14; z10 draws z12."""
    bundle = _make_bundle(start_zoom=10)
    leaf = bundle.lineage[-1]
    z14 = next(r for r in bundle.lineage if r.zoom == 14)
    z12 = next(r for r in bundle.lineage if r.zoom == 12)

    assert bundle.visuals[VisualRole.Z14_PARENT].overlay_footprint_bbox == leaf.bbox
    assert bundle.visuals[VisualRole.Z12_GRANDPARENT].overlay_footprint_bbox == z14.bbox
    assert bundle.visuals[VisualRole.Z10_GREAT_GRANDPARENT].overlay_footprint_bbox == z12.bbox


def test_bundle_json_roundtrip_preserves_keyed_surfaces():
    bundle = _make_bundle(start_zoom=12)
    s = bundle.model_dump_json()
    restored = Z16ContextBundle.model_validate_json(s)
    assert restored == bundle
    raw = json.loads(s)
    assert isinstance(raw["visuals"], dict)
    assert isinstance(raw["contexts"], dict)
    # Enum keys serialize as their string values
    assert "z14_parent" in raw["visuals"]
    assert "z15_same_center" in raw["visuals"]


def test_bundle_evidence_defaults_to_empty_dict():
    bundle = _make_bundle(start_zoom=12)
    assert bundle.evidence == {}


def test_bundle_schema_version_defaults_to_one_dot_zero():
    bundle = _make_bundle(start_zoom=12)
    assert bundle.schema_version == "1.0"
