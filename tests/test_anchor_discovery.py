"""Tests for the Phase C v1 anchor discovery pipeline.

Covers:
  - AnchorDiscoveryConfig defaults
  - assemble_anchor_structures: strict anchor_id matching, per-anchor failure
    policies (out-of-bounds, low confidence, missing coord-gen response),
    phase_history population, provenance shape
  - PLAN_CAPTURE: skipped when placement is invalid; produced when valid

LLM calls are not exercised here (run_anchor_discovery is async). The unit
tests construct the discover_meta / coords_meta dicts directly so we can
test the assembly + failure-handling logic deterministically.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from readwater.pipeline.structure.anchor_discovery import (
    AnchorDiscoveryConfig,
    AnchorDiscoveryInputs,
    assemble_anchor_structures,
    ensure_grid_overlay,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _bundle_stub() -> MagicMock:
    """assemble_anchor_structures never touches inputs.bundle — a Mock is enough."""
    return MagicMock(name="Z16ContextBundle")


def _inputs(tmp_path: Path, cell_id: str = "test-cell") -> AnchorDiscoveryInputs:
    """Inputs with stub paths — assembly tests don't read the images."""
    return AnchorDiscoveryInputs(
        cell_id=cell_id,
        bundle=_bundle_stub(),  # type: ignore[arg-type]
        z16_image_path=tmp_path / "z16.png",
        z16_grid_overlay_path=tmp_path / "z16_grid_8x8.png",
        overlay_z15_path=tmp_path / "z15.png",
        overlay_z14_path=tmp_path / "z14.png",
        overlay_z12_path=tmp_path / "z12.png",
        z16_center_latlon=(26.011172, -81.753546),  # root-10-8 center
        coverage_miles=0.37,
        area_root=tmp_path,
    )


def _discover_meta(file_mode: str = "nogrid") -> dict:
    return {
        "stage": "DISCOVER",
        "file_mode": file_mode,
        "inject_evidence": False,
        "z16_image_used": "/data/z16.png",
        "z15_image_used": "/data/z15.png",
        "z14_image_used": "/data/z14.png",
        "z12_image_used": "/data/z12.png",
        "prompt_id": "anchor_identification_v3",
        "prompt_version": f"v3_{file_mode}",
        "input_hash": "abc123",
        "ts": "2026-04-25T10:00:00Z",
    }


def _coords_meta(file_mode: str = "grid") -> dict:
    return {
        "stage": "COORDS",
        "file_mode": file_mode,
        "image_used": f"/data/z16{'_grid_8x8' if file_mode == 'grid' else ''}.png",
        "prompt_id": "anchor_coords",
        "prompt_version": f"coords_{file_mode}",
        "input_hash": "def456",
        "ts": "2026-04-25T10:01:00Z",
    }


def _v3_anchor(aid: str, **kwargs) -> dict:
    return {
        "anchor_id": aid,
        "structure_type": kwargs.get("structure_type", "island"),
        "scale": kwargs.get("scale", "major"),
        "tier": kwargs.get("tier", 1),
        "position_in_zone": kwargs.get("position_in_zone", "center"),
        "rationale": kwargs.get("rationale", "test anchor"),
        "confidence": kwargs.get("confidence", 0.85),
        "zone_id": kwargs.get("zone_id"),
    }


def _coord_anchor(aid: str, **kwargs) -> dict:
    return {
        "anchor_id": aid,
        "pixel_center": kwargs.get("pixel_center", [640, 640]),
        "pixel_bbox": kwargs.get("pixel_bbox", [600, 600, 680, 680]),
        "placement_confidence": kwargs.get("placement_confidence", 0.8),
        "placement_notes": kwargs.get("placement_notes", "centered placement"),
    }


# ----------------------------------------------------------------------
# AnchorDiscoveryConfig
# ----------------------------------------------------------------------


def test_config_defaults():
    cfg = AnchorDiscoveryConfig()
    assert cfg.v3_mode == "nogrid"
    assert cfg.coords_mode == "grid"
    assert cfg.v3_comparison_winner == "nogrid"
    assert cfg.coords_comparison_winner == "grid"
    assert cfg.inject_evidence is False
    assert cfg.tile_budget_z18 == 25


def test_config_is_frozen():
    cfg = AnchorDiscoveryConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.v3_mode = "grid"  # type: ignore[misc]


# ----------------------------------------------------------------------
# Strict anchor_id matching
# ----------------------------------------------------------------------


def test_assemble_strict_match_unmatched_coord_id_yields_finding(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [
        _coord_anchor("a1"),
        _coord_anchor("phantom_a99"),  # not in v3
    ]
    structures, cell_findings = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    # phantom_a99 → cell-level Finding, not silently ignored
    codes = [f.issue_code for f in cell_findings]
    assert "COORDS_UNMATCHED_ANCHOR_ID" in codes
    # The real a1 still produces an AnchorStructure
    assert len(structures) == 1
    assert structures[0].anchor_id == "a1"


def test_assemble_v3_anchor_with_no_coord_response(tmp_path):
    """Per-anchor failure: keep + Finding, seed_z18_fetch_plan=None."""
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1"), _v3_anchor("a2")]
    cg = [_coord_anchor("a1")]  # a2 missing from coord-gen
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    by_id = {s.anchor_id: s for s in structures}
    assert by_id["a1"].seed_z18_fetch_plan is not None
    a2 = by_id["a2"]
    assert a2.seed_z18_fetch_plan is None
    codes = [f.issue_code for f in a2.findings]
    assert "NO_COORD_RESPONSE" in codes


# ----------------------------------------------------------------------
# Out-of-bounds pixel handling
# ----------------------------------------------------------------------


def test_assemble_out_of_bounds_pixel_attaches_finding(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [_coord_anchor("a1", pixel_center=[1500, 700])]  # x=1500 outside [0, 1280)
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    a1 = structures[0]
    codes = [f.issue_code for f in a1.findings]
    assert "COORDS_OUT_OF_BOUNDS" in codes
    # Plan capture must be skipped for invalid placements
    assert a1.seed_z18_fetch_plan is None


def test_assemble_low_confidence_attaches_info_finding(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [_coord_anchor("a1", placement_confidence=0.15)]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    a1 = structures[0]
    codes_severities = [(f.issue_code, f.severity) for f in a1.findings]
    assert ("LOW_CONFIDENCE", "info") in codes_severities
    # Low confidence does NOT invalidate the placement — the plan still gets built
    assert a1.seed_z18_fetch_plan is not None


# ----------------------------------------------------------------------
# Phase history + provenance
# ----------------------------------------------------------------------


def test_assemble_phase_history_three_events_in_order(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [_coord_anchor("a1")]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    history = structures[0].phase_history
    assert [e.phase for e in history] == ["C.DISCOVER", "C.COORDS", "C.PLAN_CAPTURE"]
    assert history[0].action == "emit"
    assert history[1].action == "locate"
    assert history[2].action == "plan"


def test_assemble_provenance_records_winners(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [_coord_anchor("a1")]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg,
        _discover_meta("nogrid"), _coords_meta("grid"),
    )
    prov = structures[0].provenance
    assert prov.prompt_version == "v3_nogrid+coords_grid"
    assert prov.prompt_id == "anchor_identification_v3"
    assert prov.input_hash != ""
    assert prov.provider_config.get("model")
    # When coords_mode is grid, the grid overlay path is in overlay_refs
    assert any(p.endswith("z16_grid_8x8.png") for p in prov.overlay_refs)


def test_assemble_state_default_is_draft(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1")]
    cg = [_coord_anchor("a1")]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    assert structures[0].state == "draft"


# ----------------------------------------------------------------------
# PLAN_CAPTURE
# ----------------------------------------------------------------------


def test_assemble_plan_capture_populated_for_valid_placement(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig(tile_budget_z18=9)
    v3 = [_v3_anchor("a1")]
    # A 200-px bbox on a z16 image at lat 26 ≈ 215 m extent
    cg = [_coord_anchor("a1", pixel_bbox=[440, 440, 640, 640], pixel_center=[540, 540])]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    plan = structures[0].seed_z18_fetch_plan
    assert plan is not None
    assert plan.tile_budget == 9
    # 215m × 1.25 padding ≈ 269m; one z18 tile at lat 26 covers ~343m, so 1×1
    assert len(plan.tile_centers) >= 1
    assert plan.extent_meters > 0


def test_assemble_priority_rank_follows_v3_order(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1"), _v3_anchor("a2"), _v3_anchor("a3")]
    cg = [_coord_anchor("a1"), _coord_anchor("a2"), _coord_anchor("a3")]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    by_id = {s.anchor_id: s.priority_rank for s in structures}
    assert by_id == {"a1": 1, "a2": 2, "a3": 3}


def test_assemble_zone_id_propagates(tmp_path):
    inputs = _inputs(tmp_path)
    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1", zone_id="z2")]
    cg = [_coord_anchor("a1")]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )
    assert structures[0].zone_id == "z2"


# ----------------------------------------------------------------------
# ensure_grid_overlay
# ----------------------------------------------------------------------


def test_ensure_grid_overlay_rejects_4x4_legacy_cache(tmp_path):
    """A cached file with mismatched dims should be regenerated, not trusted."""
    from PIL import Image

    src = tmp_path / "z16.png"
    Image.new("RGB", (1280, 1280), (50, 100, 50)).save(src)

    # Pretend a legacy 4x4 cache was written at a different resolution.
    cache = tmp_path / "z16_grid_8x8.png"
    Image.new("RGB", (640, 640), (200, 200, 200)).save(cache)
    assert cache.stat().st_size > 0

    # Should regenerate (cache dims don't match source)
    out = ensure_grid_overlay(src, cache)
    assert out == cache
    with Image.open(cache) as fresh:
        assert fresh.size == (1280, 1280)


def test_ensure_grid_overlay_uses_cache_when_dims_match(tmp_path):
    """When the cache matches source dimensions, skip rendering (mtime unchanged)."""
    from PIL import Image

    src = tmp_path / "z16.png"
    Image.new("RGB", (1280, 1280), (50, 100, 50)).save(src)

    cache = tmp_path / "z16_grid_8x8.png"
    out1 = ensure_grid_overlay(src, cache)
    mtime1 = out1.stat().st_mtime

    # Second call — should return same path without re-rendering
    out2 = ensure_grid_overlay(src, cache)
    assert out2.stat().st_mtime == mtime1


# ----------------------------------------------------------------------
# Review overlay rendering (production-side artifact)
# ----------------------------------------------------------------------


def test_render_review_overlay_writes_png(tmp_path):
    """Production renderer should produce a PNG at <output_dir>/<cell>_review_overlay.png."""
    from PIL import Image

    from readwater.pipeline.structure.anchor_discovery import _render_review_overlay

    inputs = _inputs(tmp_path, cell_id="root-test")
    Image.new("RGB", (1280, 1280), (50, 100, 50)).save(inputs.z16_image_path)

    cfg = AnchorDiscoveryConfig()
    v3 = [_v3_anchor("a1"), _v3_anchor("a2")]
    cg = [_coord_anchor("a1"), _coord_anchor("a2", pixel_center=[200, 200],
                                              pixel_bbox=[150, 150, 250, 250])]
    structures, _ = assemble_anchor_structures(
        inputs, cfg, v3, cg, _discover_meta(), _coords_meta(),
    )

    out = _render_review_overlay(inputs, structures, tmp_path)
    assert out is not None
    assert out.name == "root-test_review_overlay.png"
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.size == (1280, 1280)


def test_render_review_overlay_skips_gracefully_when_z16_missing(tmp_path):
    """Missing z16 image must not raise — overlay is auxiliary."""
    from readwater.pipeline.structure.anchor_discovery import _render_review_overlay

    inputs = _inputs(tmp_path)  # z16 path doesn't exist
    out = _render_review_overlay(inputs, [], tmp_path)
    assert out is None


def test_config_render_review_overlay_default_on():
    """The default config opts in to overlay rendering."""
    cfg = AnchorDiscoveryConfig()
    assert cfg.render_review_overlay is True
