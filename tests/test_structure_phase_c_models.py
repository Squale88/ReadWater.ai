"""Tests for the Phase C v1 envelope additions to AnchorStructure.

Covers PhaseEvent, Provenance, Finding, Z18FetchPlan, and the new fields on
AnchorStructure (state, phase_history, provenance, findings,
seed_z18_fetch_plan, priority_rank, zone_id).

Migration policy from docs/PHASE_C_TASKS.md addendum: legacy cached anchor
JSON gets a backfilled Provenance (prompt_version="legacy_pre_v1") rather
than silently defaulting to empty. The required-field enforcement test
below is what makes that policy load-bearing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from readwater.models.structure import (
    AnchorStructure,
    Finding,
    ObservedGeometry,
    PhaseEvent,
    Provenance,
    Z18FetchPlan,
)


# --- shared fixtures ---


def _geom() -> ObservedGeometry:
    return ObservedGeometry(
        pixel_polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        latlon_polygon=[
            (26.01, -81.75), (26.01, -81.74),
            (26.00, -81.74), (26.00, -81.75),
        ],
        image_ref="z0_10_8.png",
        extractor="gridcell",
        extraction_mode="region",
    )


def _prov(version: str = "v3_nogrid") -> Provenance:
    return Provenance(prompt_id="anchor_identification_v3", prompt_version=version)


# --- PhaseEvent ---


def test_phase_event_required_fields():
    evt = PhaseEvent(
        phase="C.DISCOVER",
        action="emit",
        actor="structure.agent",
        timestamp="2026-04-23T12:00:00Z",
    )
    assert evt.phase == "C.DISCOVER"
    assert evt.note is None


def test_phase_event_with_note():
    evt = PhaseEvent(
        phase="C.COORDS",
        action="update",
        actor="coord_gen",
        timestamp="2026-04-23T12:01:00Z",
        note="Pixel center 482,143 -> latlon (26.0160, -81.7552)",
    )
    assert "482,143" in evt.note


# --- Provenance ---


def test_provenance_minimal():
    p = Provenance(prompt_id="foo", prompt_version="v3_nogrid")
    assert p.source_images == []
    assert p.overlay_refs == []
    assert p.provider_config == {}
    assert p.input_hash == ""


def test_provenance_full():
    p = Provenance(
        source_images=["img1.png", "img2.png"],
        overlay_refs=["grid.png"],
        prompt_id="anchor_identification",
        prompt_version="v3_grid",
        provider_config={"model": "claude-sonnet-4-5", "temperature": 0.0},
        input_hash="abc123",
    )
    assert p.provider_config["model"] == "claude-sonnet-4-5"


def test_provenance_legacy_marker_round_trips():
    """Per addendum: legacy cached anchors get a backfilled marker."""
    p = Provenance(prompt_id="unknown", prompt_version="legacy_pre_v1")
    dumped = p.model_dump()
    restored = Provenance.model_validate(dumped)
    assert restored.prompt_version == "legacy_pre_v1"


# --- Finding ---


def test_finding_severity_validated():
    f = Finding(
        issue_code="COORDS_OUT_OF_BOUNDS",
        severity="warn",
        object_id="a1",
        message="pixel_center (1500, 700) outside image bounds (1280, 1280)",
    )
    assert f.severity == "warn"
    assert f.field is None
    assert f.recommended_action is None


def test_finding_invalid_severity():
    with pytest.raises(ValidationError):
        Finding(
            issue_code="X",
            severity="critical",  # not in {info, warn, error}
            object_id="a1",
            message="boom",
        )


def test_finding_with_field_and_action():
    f = Finding(
        issue_code="LOW_CONFIDENCE",
        severity="info",
        object_id="a3",
        field="placement_confidence",
        message="placement_confidence=0.2 is low",
        recommended_action="rerun coord-gen with stronger prompt",
    )
    assert f.field == "placement_confidence"
    assert "rerun" in f.recommended_action


# --- Z18FetchPlan ---


def test_z18_fetch_plan_defaults():
    plan = Z18FetchPlan()
    assert plan.tile_centers == []
    assert plan.tile_budget == 25
    assert plan.extent_meters == 0.0


def test_z18_fetch_plan_populated():
    plan = Z18FetchPlan(
        tile_centers=[(26.01, -81.75), (26.01, -81.74), (26.00, -81.75)],
        tile_budget=9,
        extent_meters=200.0,
    )
    assert len(plan.tile_centers) == 3
    assert plan.tile_budget == 9


# --- AnchorStructure with Phase C v1 fields ---


def test_anchor_requires_provenance():
    """Required-field enforcement is what makes the migration policy
    meaningful: legacy data must be migrated, not silently defaulted."""
    with pytest.raises(ValidationError):
        AnchorStructure(
            anchor_id="a1",
            structure_type="island",
            scale="major",
            anchor_center_latlon=(26.01, -81.75),
            geometry=_geom(),
            confidence=0.7,
            # provenance intentionally missing
        )


def test_anchor_default_phase_c_fields():
    a = AnchorStructure(
        anchor_id="a1",
        structure_type="island",
        scale="major",
        anchor_center_latlon=(26.01, -81.75),
        geometry=_geom(),
        confidence=0.7,
        provenance=_prov(),
    )
    assert a.state == "draft"
    assert a.phase_history == []
    assert a.findings == []
    assert a.seed_z18_fetch_plan is None
    assert a.priority_rank is None
    assert a.zone_id is None


def test_anchor_state_literal_validated():
    with pytest.raises(ValidationError):
        AnchorStructure(
            anchor_id="a1",
            structure_type="island",
            scale="major",
            anchor_center_latlon=(26.01, -81.75),
            geometry=_geom(),
            confidence=0.7,
            provenance=_prov(),
            state="something_else",  # not in {draft, validated, approved, rejected}
        )


def test_anchor_full_round_trip():
    """Building an anchor with every Phase C field, dumping, and reloading
    must reconstruct the same object."""
    plan = Z18FetchPlan(
        tile_centers=[(26.011, -81.754), (26.012, -81.753)],
        tile_budget=9,
        extent_meters=180.0,
    )
    history = [
        PhaseEvent(phase="C.DISCOVER", action="emit", actor="v3_nogrid",
                   timestamp="2026-04-23T12:00:00Z"),
        PhaseEvent(phase="C.COORDS", action="update", actor="coord_gen",
                   timestamp="2026-04-23T12:01:00Z", note="placed at 482,143"),
    ]
    findings = [
        Finding(issue_code="OK", severity="info", object_id="a1",
                message="placement looks clean"),
    ]
    a = AnchorStructure(
        anchor_id="a1",
        structure_type="drain_system",
        scale="major",
        anchor_center_latlon=(26.0160, -81.7552),
        geometry=_geom(),
        confidence=0.8,
        provenance=_prov("v3_grid"),
        state="validated",
        phase_history=history,
        findings=findings,
        seed_z18_fetch_plan=plan,
        priority_rank=1,
        zone_id="z2",
    )

    dumped = a.model_dump()
    restored = AnchorStructure.model_validate(dumped)

    assert restored.anchor_id == "a1"
    assert restored.state == "validated"
    assert len(restored.phase_history) == 2
    assert restored.phase_history[1].note == "placed at 482,143"
    assert restored.seed_z18_fetch_plan.tile_budget == 9
    assert restored.priority_rank == 1
    assert restored.zone_id == "z2"
    assert restored.provenance.prompt_version == "v3_grid"


def test_anchor_extra_fields_allowed_for_forward_compat():
    """extra='allow' lets older snapshots load even if they had extra keys."""
    a = AnchorStructure(
        anchor_id="a1",
        structure_type="island",
        scale="major",
        anchor_center_latlon=(26.01, -81.75),
        geometry=_geom(),
        confidence=0.7,
        provenance=_prov(),
        legacy_field="ignored value",  # not in the schema
    )
    dumped = a.model_dump()
    assert dumped.get("legacy_field") == "ignored value"
