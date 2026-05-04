"""Tests for cell analyzer coordinate geometry, ground coverage, and filenames.

Reference point: Marco Island, FL — center (25.94, -81.73).
All grid tests use sections=4 (4x4 = 16 sub-cells).
"""

import math

from readwater.pipeline.cell_analyzer import (
    MILES_PER_DEG_LAT,
    SECTIONS,
    _image_filename,
    _make_bbox,
    _make_cell_id,
    _miles_per_deg_lon,
    _role_for_zoom,
    _sub_cell_bbox,
    _subdivide_bbox,
    ground_coverage_miles,
)

MARCO = (25.94, -81.73)
MARCO_LAT = 25.94


# --- Longitude conversion ---


def test_miles_per_deg_lon_equator():
    assert abs(_miles_per_deg_lon(0.0) - MILES_PER_DEG_LAT) < 0.01


def test_miles_per_deg_lon_poles():
    assert abs(_miles_per_deg_lon(90.0)) < 0.01


def test_miles_per_deg_lon_marco_island():
    expected = MILES_PER_DEG_LAT * math.cos(math.radians(MARCO_LAT))
    assert abs(_miles_per_deg_lon(MARCO_LAT) - expected) < 0.001


# --- Ground coverage ---


def test_ground_coverage_zoom_10():
    cov = ground_coverage_miles(10, MARCO_LAT)
    # 2.5 * 24901 * cos(25.94) / 1024 ≈ 54.7
    assert 53 < cov < 56


def test_ground_coverage_zoom_12():
    cov = ground_coverage_miles(12, MARCO_LAT)
    assert 13 < cov < 15


def test_ground_coverage_zoom_14():
    cov = ground_coverage_miles(14, MARCO_LAT)
    assert 3 < cov < 4


def test_ground_coverage_zoom_16():
    cov = ground_coverage_miles(16, MARCO_LAT)
    assert 0.8 < cov < 1.0


def test_ground_coverage_zoom_18():
    cov = ground_coverage_miles(18, MARCO_LAT)
    assert 0.2 < cov < 0.25


def test_ground_coverage_monotonically_decreasing():
    zooms = [10, 12, 14, 16, 18]
    coverages = [ground_coverage_miles(z, MARCO_LAT) for z in zooms]
    for i in range(len(coverages) - 1):
        assert coverages[i] > coverages[i + 1]


# --- Bounding box ---


def test_bbox_center_matches_input():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lat, lon = bbox.center
    assert abs(lat - MARCO[0]) < 1e-10
    assert abs(lon - MARCO[1]) < 1e-10


def test_bbox_lat_span_matches_size():
    size = ground_coverage_miles(12, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lat_span_miles = (bbox.north - bbox.south) * MILES_PER_DEG_LAT
    assert abs(lat_span_miles - size) < 0.01


def test_bbox_lon_span_matches_size():
    size = ground_coverage_miles(12, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    lon_span_miles = (bbox.east - bbox.west) * _miles_per_deg_lon(MARCO_LAT)
    assert abs(lon_span_miles - size) < 0.01


def test_bbox_is_wider_in_degrees_than_tall():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert (bbox.east - bbox.west) > (bbox.north - bbox.south)


def test_bbox_north_south_symmetric():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert abs((bbox.north - MARCO[0]) - (MARCO[0] - bbox.south)) < 1e-10


def test_bbox_east_west_symmetric():
    size = ground_coverage_miles(10, MARCO_LAT)
    bbox = _make_bbox(MARCO, size)
    assert abs((bbox.east - MARCO[1]) - (MARCO[1] - bbox.west)) < 1e-10


# --- 4x4 grid subdivision ---


def _marco_bbox():
    size = ground_coverage_miles(10, MARCO_LAT)
    return _make_bbox(MARCO, size)


def test_subdivide_4x4_returns_16():
    assert len(_subdivide_bbox(_marco_bbox(), SECTIONS)) == 16


def test_subdivide_4x4_row_col_coverage():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    positions = {(r, c) for r, c, _ in subs}
    expected = {(r, c) for r in range(4) for c in range(4)}
    assert positions == expected


def test_subdivide_4x4_centers_within_parent():
    bbox = _marco_bbox()
    for row, col, (lat, lon) in _subdivide_bbox(bbox, SECTIONS):
        assert bbox.south < lat < bbox.north
        assert bbox.west < lon < bbox.east


def test_subdivide_4x4_centers_ordered_north_to_south():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    by_row = {}
    for row, col, (lat, _) in subs:
        by_row.setdefault(row, []).append(lat)
    for row in range(SECTIONS - 1):
        assert min(by_row[row]) > max(by_row[row + 1])


def test_subdivide_4x4_centers_ordered_west_to_east():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    by_col = {}
    for row, col, (_, lon) in subs:
        by_col.setdefault(col, []).append(lon)
    for col in range(SECTIONS - 1):
        assert max(by_col[col]) < min(by_col[col + 1])


def test_subdivide_4x4_centers_evenly_spaced_lat():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    lats = sorted({lat for _, _, (lat, _) in subs}, reverse=True)
    assert len(lats) == SECTIONS
    spacing = lats[0] - lats[1]
    for i in range(len(lats) - 1):
        assert abs((lats[i] - lats[i + 1]) - spacing) < 1e-10


def test_subdivide_4x4_centers_evenly_spaced_lon():
    subs = _subdivide_bbox(_marco_bbox(), SECTIONS)
    lons = sorted({lon for _, _, (_, lon) in subs})
    assert len(lons) == SECTIONS
    spacing = lons[1] - lons[0]
    for i in range(len(lons) - 1):
        assert abs((lons[i + 1] - lons[i]) - spacing) < 1e-10


# --- Sub-cell bounding boxes: tiling ---


def test_sub_cell_4x4_bboxes_tile_parent():
    bbox = _marco_bbox()
    norths, souths, easts, wests = [], [], [], []
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            norths.append(sb.north)
            souths.append(sb.south)
            easts.append(sb.east)
            wests.append(sb.west)
    assert abs(max(norths) - bbox.north) < 1e-10
    assert abs(min(souths) - bbox.south) < 1e-10
    assert abs(max(easts) - bbox.east) < 1e-10
    assert abs(min(wests) - bbox.west) < 1e-10


def test_sub_cell_4x4_no_gaps_horizontal():
    bbox = _marco_bbox()
    for row in range(SECTIONS):
        for col in range(SECTIONS - 1):
            left = _sub_cell_bbox(bbox, row, col, SECTIONS)
            right = _sub_cell_bbox(bbox, row, col + 1, SECTIONS)
            assert abs(left.east - right.west) < 1e-10


def test_sub_cell_4x4_no_gaps_vertical():
    bbox = _marco_bbox()
    for col in range(SECTIONS):
        for row in range(SECTIONS - 1):
            upper = _sub_cell_bbox(bbox, row, col, SECTIONS)
            lower = _sub_cell_bbox(bbox, row + 1, col, SECTIONS)
            assert abs(upper.south - lower.north) < 1e-10


def test_sub_cell_4x4_center_matches_subdivide():
    bbox = _marco_bbox()
    subs = {(r, c): center for r, c, center in _subdivide_bbox(bbox, SECTIONS)}
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            assert abs(sb.center[0] - subs[(row, col)][0]) < 1e-10
            assert abs(sb.center[1] - subs[(row, col)][1]) < 1e-10


def test_sub_cell_4x4_size_is_quarter_of_parent():
    bbox = _marco_bbox()
    parent_h = bbox.north - bbox.south
    parent_w = bbox.east - bbox.west
    for row in range(SECTIONS):
        for col in range(SECTIONS):
            sb = _sub_cell_bbox(bbox, row, col, SECTIONS)
            assert abs((sb.north - sb.south) - parent_h / SECTIONS) < 1e-10
            assert abs((sb.east - sb.west) - parent_w / SECTIONS) < 1e-10


def test_two_level_4x4_subdivision_tiles():
    """Subdivide a sub-cell again — 16 grandchild bboxes should tile it."""
    parent = _marco_bbox()
    child = _sub_cell_bbox(parent, 1, 2, SECTIONS)
    for row in range(SECTIONS):
        for col in range(SECTIONS - 1):
            left = _sub_cell_bbox(child, row, col, SECTIONS)
            right = _sub_cell_bbox(child, row, col + 1, SECTIONS)
            assert abs(left.east - right.west) < 1e-10
    for col in range(SECTIONS):
        for row in range(SECTIONS - 1):
            upper = _sub_cell_bbox(child, row, col, SECTIONS)
            lower = _sub_cell_bbox(child, row + 1, col, SECTIONS)
            assert abs(upper.south - lower.north) < 1e-10


# --- Cell ID generation ---


def test_make_cell_id_root():
    assert _make_cell_id(None, 0, 0) == "root"


def test_make_cell_id_child():
    # row 1, col 2 in 4x4 grid = cell 7
    assert _make_cell_id("root", 1, 2) == "root-7"


def test_make_cell_id_nested():
    # row 0, col 1 in 4x4 grid = cell 2
    assert _make_cell_id("root-7", 0, 1) == "root-7-2"


# --- Image filename ---


def test_image_filename_root():
    assert _image_filename("root", 0) == "z0.png"


def test_image_filename_root_with_provider():
    assert _image_filename("root", 0, "google_static") == "z0_google_static.png"


def test_image_filename_depth1():
    assert _image_filename("root-14", 1) == "z0_14.png"


def test_image_filename_depth2_with_provider():
    assert _image_filename("root-14-3", 2, "naip") == "z0_14_3_naip.png"


def test_image_filename_cell_16():
    """Cell 16 (row 3, col 3) is valid in a 4x4 grid."""
    assert _image_filename("root-16", 1) == "z0_16.png"


# --- Role for zoom ---


def test_role_overview_at_low_zoom():
    for z in [10, 12, 14, 15]:
        assert _role_for_zoom(z) == "overview"


def test_role_structure_at_high_zoom():
    for z in [16, 18, 20]:
        assert _role_for_zoom(z) == "structure"


# --- Step 7: ancestor lineage propagation ---


from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from readwater.api.providers.placeholder import PlaceholderProvider  # noqa: E402
from readwater.api.providers.registry import ImageProviderRegistry  # noqa: E402
from readwater.pipeline import cell_analyzer  # noqa: E402


def _make_dry_run_registry():
    registry = ImageProviderRegistry()
    p = PlaceholderProvider(size=64)
    registry.register(p, roles=["overview", "structure"])
    return registry


async def _dry_run_with_spy(tmp_path, max_depth=2, start_zoom=12):
    """Drive analyze_cell in dry_run mode and capture every recursive call's
    ancestor_lineage + position_in_parent via a module-level spy."""
    registry = _make_dry_run_registry()
    real = cell_analyzer._analyze_recursive
    captured: list[dict] = []

    async def spy(*args, **kwargs):
        captured.append({
            "cell_id": kwargs.get("cell_id"),
            "depth": kwargs.get("depth"),
            "zoom": kwargs.get("zoom"),
            "ancestor_lineage_ids": [
                r.cell_id for r in (kwargs.get("ancestor_lineage") or [])
            ],
            "ancestor_lineage_zooms": [
                r.zoom for r in (kwargs.get("ancestor_lineage") or [])
            ],
            "position_in_parent": kwargs.get("position_in_parent"),
            "ancestor_contexts_keys": list((kwargs.get("ancestor_contexts") or {}).keys()),
        })
        return await real(*args, **kwargs)

    with patch.object(cell_analyzer, "_analyze_recursive", new=spy):
        await cell_analyzer.analyze_cell(
            center=MARCO,
            registry=registry,
            start_zoom=start_zoom,
            threshold=4.0,
            max_depth=max_depth,
            dry_run=True,
            output_dir=str(tmp_path),
        )
    return captured


@pytest.mark.asyncio
async def test_recursion_root_has_empty_lineage(tmp_path):
    captured = await _dry_run_with_spy(tmp_path, max_depth=1, start_zoom=12)
    root = next(c for c in captured if c["cell_id"] == "root")
    assert root["ancestor_lineage_ids"] == []
    assert root["ancestor_contexts_keys"] == []
    assert root["position_in_parent"] is None


@pytest.mark.asyncio
async def test_recursion_depth1_children_carry_root_lineage(tmp_path):
    captured = await _dry_run_with_spy(tmp_path, max_depth=1, start_zoom=12)
    d1 = [c for c in captured if c["depth"] == 1]
    assert len(d1) == 16
    for entry in d1:
        # Each child sees the root as its only ancestor.
        assert entry["ancestor_lineage_ids"] == ["root"]
        assert entry["ancestor_lineage_zooms"] == [12]
        # Each child knows its own position in the parent.
        row, col = entry["position_in_parent"]
        assert 0 <= row < 4
        assert 0 <= col < 4


@pytest.mark.asyncio
async def test_recursion_depth2_grandchildren_carry_two_ancestors(tmp_path):
    captured = await _dry_run_with_spy(tmp_path, max_depth=2, start_zoom=12)
    d2 = [c for c in captured if c["depth"] == 2]
    # Full fanout in dry_run (all scores=5): 16 * 16 = 256.
    assert len(d2) == 256
    for entry in d2:
        ids = entry["ancestor_lineage_ids"]
        zooms = entry["ancestor_lineage_zooms"]
        # Exactly two ancestors, root first then the z14 parent.
        assert len(ids) == 2
        assert ids[0] == "root"
        assert ids[1].startswith("root-")
        assert zooms == [12, 14]
        assert entry["position_in_parent"] is not None


@pytest.mark.asyncio
async def test_recursion_position_in_parent_matches_cell_id_suffix(tmp_path):
    """The position_in_parent (row, col) must match the cell-number suffix
    of the child cell_id. cell_num = row*4 + col + 1."""
    captured = await _dry_run_with_spy(tmp_path, max_depth=1, start_zoom=12)
    for entry in (c for c in captured if c["depth"] == 1):
        suffix = int(entry["cell_id"].rsplit("-", 1)[1])
        row, col = entry["position_in_parent"]
        assert row * 4 + col + 1 == suffix


@pytest.mark.asyncio
async def test_recursion_ancestor_contexts_is_empty_dict_in_step_7(tmp_path):
    """Step 7 carries ancestor_contexts through but does not populate it.
    Step 8 will fill it in."""
    captured = await _dry_run_with_spy(tmp_path, max_depth=2, start_zoom=12)
    for entry in captured:
        assert entry["ancestor_contexts_keys"] == []


# --- Step 8: build_cell_context invoked on each retained cell ---


from readwater.models.context import CellContext  # noqa: E402
from readwater.pipeline.cv.cell_pipeline import CellResult  # noqa: E402


def _stub_grid_score_result(keep_cell_num: int = 1):
    """Grid-scoring payload that keeps exactly one sub-cell."""
    return {
        "summary": "stub",
        "hydrology_notes": "",
        "sub_scores": [
            {
                "cell_number": i,
                "score": 5 if i == keep_cell_num else 0,
                "reasoning": "stub",
            }
            for i in range(1, 17)
        ],
        "raw_response_yes": "",
        "raw_response_no": "",
    }


@pytest.fixture
def live_recursion_patches(tmp_path):
    """Install patches that let analyze_cell run non-dry-run without any
    real Claude API calls. Yields a dict of recorders the test can inspect."""

    build_calls: list[dict] = []
    structure_calls: list[dict] = []

    async def spy_build(**kwargs):
        build_calls.append({
            "cell_id": kwargs["cell_id"],
            "zoom": kwargs["zoom"],
            "ancestor_lineage_ids": [
                r.cell_id for r in (kwargs.get("ancestor_lineage") or [])
            ],
            "ancestor_context_keys": list(
                (kwargs.get("ancestor_contexts") or {}).keys()
            ),
            "grid_scoring_summary": (kwargs.get("grid_scoring_result") or {}).get("summary"),
        })
        return CellContext(cell_id=kwargs["cell_id"], zoom=kwargs["zoom"])

    async def stub_dual_pass(*args, **kwargs):
        return _stub_grid_score_result(keep_cell_num=1)

    def stub_run_cell_full(area_id, cell_id, **kwargs):
        structure_calls.append(cell_id)
        return CellResult(cell_id=cell_id, succeeded=True)

    async def stub_confirm(*args, **kwargs):
        return {"has_fishing_water": True, "raw_response": ""}

    patchers = [
        patch.object(cell_analyzer, "build_cell_context", new=spy_build),
        patch.object(cell_analyzer, "dual_pass_grid_scoring", new=stub_dual_pass),
        patch.object(cell_analyzer, "run_cell_full", new=stub_run_cell_full),
        patch.object(cell_analyzer, "confirm_fishing_water", new=stub_confirm),
        patch.object(cell_analyzer, "draw_grid_overlay", return_value=str(tmp_path / "fake_grid.png")),
    ]
    for p in patchers:
        p.start()
    try:
        yield {"build_calls": build_calls, "structure_calls": structure_calls}
    finally:
        for p in patchers:
            p.stop()


@pytest.mark.asyncio
async def test_step8_build_cell_context_called_once_per_retained_cell(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    cells = await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    # With only cell 1 retained at each step: root, root-1, root-1-1.
    ids = [c.id for c in cells]
    assert set(ids) == {"root", "root-1", "root-1-1"}

    build_ids = [c["cell_id"] for c in live_recursion_patches["build_calls"]]
    assert sorted(build_ids) == ["root", "root-1", "root-1-1"]


@pytest.mark.asyncio
async def test_step8_cell_context_stored_on_analysis(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    cells = await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    for cell in cells:
        assert cell.analysis is not None
        assert cell.analysis.context is not None
        assert cell.analysis.context.cell_id == cell.id
        assert cell.analysis.context.zoom == cell.zoom_level


@pytest.mark.asyncio
async def test_step8_ancestor_contexts_propagate_to_children(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    calls = {c["cell_id"]: c for c in live_recursion_patches["build_calls"]}

    assert calls["root"]["ancestor_context_keys"] == []
    assert calls["root-1"]["ancestor_context_keys"] == ["root"]
    assert set(calls["root-1-1"]["ancestor_context_keys"]) == {"root", "root-1"}


@pytest.mark.asyncio
async def test_step8_grid_scoring_digest_is_passed_to_build(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=0,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    root = live_recursion_patches["build_calls"][0]
    # The stub dual_pass_grid_scoring returns summary="stub".
    assert root["grid_scoring_summary"] == "stub"


# --- Step 9: Z16 bundle written to disk at handoff ---


import json  # noqa: E402
from pathlib import Path  # noqa: E402

from readwater.models.context import VisualRole, Z16ContextBundle  # noqa: E402


@pytest.mark.asyncio
async def test_step9_z16_bundle_written_at_handoff(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    # Only root-1-1 reaches z=16 in this stub run.
    bundle_path = tmp_path / "structures" / "root-1-1" / "context_bundle.json"
    assert bundle_path.exists()
    bundle = Z16ContextBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8"),
    )
    # 4 visuals for a z12-start run.
    assert set(bundle.visuals.keys()) == {
        VisualRole.Z16_LOCAL,
        VisualRole.Z15_SAME_CENTER,
        VisualRole.Z14_PARENT,
        VisualRole.Z12_GRANDPARENT,
    }
    # 3 context entries (root, root-1, root-1-1).
    assert set(bundle.contexts.keys()) == {"root", "root-1", "root-1-1"}
    # Lineage traversal order root -> parent -> self.
    assert [r.cell_id for r in bundle.lineage] == ["root", "root-1", "root-1-1"]


@pytest.mark.asyncio
async def test_step9_metadata_records_bundle_path(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()
    entries = json.loads(meta_path.read_text(encoding="utf-8"))
    z16_entry = next(e for e in entries if e["cell_id"] == "root-1-1")
    assert "context_bundle_path" in z16_entry
    assert Path(z16_entry["context_bundle_path"]).exists()
    # Non-z16 cells do NOT record a bundle path.
    for entry in entries:
        if entry["cell_id"] != "root-1-1":
            assert "context_bundle_path" not in entry


@pytest.mark.asyncio
async def test_step9_overlay_files_exist_for_ancestor_visuals(
    tmp_path, live_recursion_patches,
):
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        max_api_calls=1000,
        dry_run=False,
        output_dir=str(tmp_path),
    )
    bundle_path = tmp_path / "structures" / "root-1-1" / "context_bundle.json"
    bundle = Z16ContextBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    # Every non-local visual has a rendered overlay on disk.
    for role, ref in bundle.visuals.items():
        if role == VisualRole.Z16_LOCAL:
            assert ref.overlay_image_path is None
        else:
            assert ref.overlay_image_path is not None
            assert Path(ref.overlay_image_path).exists()


@pytest.mark.asyncio
async def test_step9_no_bundle_in_dry_run(tmp_path):
    """Dry run never reaches live image fetches -> no bundle should be produced."""
    registry = _make_dry_run_registry()
    await cell_analyzer.analyze_cell(
        center=MARCO,
        registry=registry,
        start_zoom=12,
        max_depth=2,
        dry_run=True,
        output_dir=str(tmp_path),
    )
    assert not (tmp_path / "structures").exists()


@pytest.mark.asyncio
async def test_step8_failure_does_not_abort_recursion(tmp_path):
    """If build_cell_context raises, analysis.context stays None and
    recursion continues."""
    registry = _make_dry_run_registry()

    async def failing_build(**kwargs):
        raise RuntimeError("boom")

    async def stub_dual_pass(*args, **kwargs):
        return _stub_grid_score_result(keep_cell_num=1)

    def stub_run_cell_full(area_id, cell_id, **kwargs):
        return CellResult(cell_id=cell_id, succeeded=True)

    async def stub_confirm(*args, **kwargs):
        return {"has_fishing_water": True, "raw_response": ""}

    with (
        patch.object(cell_analyzer, "build_cell_context", new=failing_build),
        patch.object(cell_analyzer, "dual_pass_grid_scoring", new=stub_dual_pass),
        patch.object(cell_analyzer, "run_cell_full", new=stub_run_cell_full),
        patch.object(cell_analyzer, "confirm_fishing_water", new=stub_confirm),
        patch.object(
            cell_analyzer, "draw_grid_overlay",
            return_value=str(tmp_path / "fake_grid.png"),
        ),
    ):
        cells = await cell_analyzer.analyze_cell(
            center=MARCO,
            registry=registry,
            start_zoom=12,
            max_depth=1,
            max_api_calls=1000,
            dry_run=False,
            output_dir=str(tmp_path),
        )
    # We still got both the root and its surviving child.
    assert {c.id for c in cells} == {"root", "root-1"}
    # Every cell's context is None because build raised.
    for c in cells:
        assert c.analysis.context is None
