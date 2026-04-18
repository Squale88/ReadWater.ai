"""Tests for the recursive cell analysis pipeline.

Uses Marco Island (25.94, -81.73) as the reference area.
All tests use PlaceholderProvider — no API keys required.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from readwater.api.providers.placeholder import PlaceholderProvider
from readwater.api.providers.registry import ImageProviderRegistry
from readwater.models.structure import StructurePhaseResult
from readwater.pipeline.cell_analyzer import (
    analyze_cell,
    ground_coverage_miles,
)


def _placeholder_structure_phase_result(cell_id: str = "test-cell") -> StructurePhaseResult:
    """Empty structure-phase result — the pipeline tests don't exercise the agent itself."""
    return StructurePhaseResult(cell_id=cell_id)

MARCO = (25.94, -81.73)
MARCO_LAT = 25.94


# --- Fixtures ---


@pytest.fixture
def registry():
    """Single PlaceholderProvider registered for both roles."""
    p = PlaceholderProvider()
    reg = ImageProviderRegistry()
    reg.register(p, ["overview", "structure"])
    return reg


def _placeholder_grid_result():
    """Default grid analysis result: all cells score 5."""
    return {
        "summary": "Placeholder grid analysis",
        "sub_scores": [
            {"cell_number": i, "score": 5.0, "reasoning": "Placeholder"}
            for i in range(1, 17)
        ],
        "hydrology_notes": "",
        "raw_response": "",
    }


def _placeholder_structure_result():
    """Default structure analysis result."""
    return {
        "summary": "Placeholder structure analysis",
        "fishable_features": [],
        "tide_interaction": "",
        "wind_exposure": "",
        "recommended_species": [],
        "access_notes": "",
        "overall_rating": 5.0,
        "raw_response": "",
    }


def _placeholder_confirm_result():
    """Default confirmation result: always confirms fishing water."""
    return {"has_fishing_water": True, "reasoning": "Placeholder", "raw_response": ""}


@pytest.fixture
def fast_registry(registry):
    """Registry with asyncio.sleep and Claude vision mocked out."""
    with (
        patch("readwater.pipeline.cell_analyzer.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "readwater.pipeline.cell_analyzer.analyze_grid_image",
            new_callable=AsyncMock,
            return_value=_placeholder_grid_result(),
        ),
        patch(
            "readwater.pipeline.cell_analyzer.run_structure_phase",
            new_callable=AsyncMock,
            return_value=_placeholder_structure_phase_result(),
        ),
        patch(
            "readwater.pipeline.cell_analyzer.confirm_fishing_water",
            new_callable=AsyncMock,
            return_value=_placeholder_confirm_result(),
        ),
        patch(
            "readwater.pipeline.cell_analyzer.draw_grid_overlay",
            side_effect=lambda p, **kw: p,
        ),
    ):
        yield registry


# --- Dry run: tree structure ---


async def test_dry_run_cell_count_depth2(registry):
    """1 + 16 + 256 = 273 cells with 4x4 grids and max_depth=2."""
    cells = await analyze_cell(MARCO, registry, max_depth=2, dry_run=True)
    assert len(cells) == 273


async def test_dry_run_no_images(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    for c in cells:
        assert c.image_path is None
        assert c.provider_images == {}


async def test_dry_run_root_cell(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    root = cells[0]
    assert root.id == "root"
    assert root.parent_id is None
    assert root.depth == 0
    assert root.zoom_level == 10
    assert abs(root.center[0] - MARCO[0]) < 1e-10
    assert abs(root.center[1] - MARCO[1]) < 1e-10


async def test_dry_run_root_has_16_children(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    assert len(cells[0].children_ids) == 16


async def test_dry_run_level_counts(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=2, dry_run=True)
    by_depth = {}
    for c in cells:
        by_depth.setdefault(c.depth, []).append(c)
    assert len(by_depth[0]) == 1
    assert len(by_depth[1]) == 16
    assert len(by_depth[2]) == 256


async def test_dry_run_parent_ids(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=2, dry_run=True)
    cell_map = {c.id: c for c in cells}
    depth_1_ids = {c.id for c in cells if c.depth == 1}

    for c in cells:
        if c.depth == 1:
            assert c.parent_id == "root"
        elif c.depth == 2:
            assert c.parent_id in depth_1_ids
            assert c.id in cell_map[c.parent_id].children_ids


async def test_dry_run_zoom_progression(registry):
    """Zoom steps by 2 per depth: 10 -> 12 -> 14."""
    cells = await analyze_cell(MARCO, registry, start_zoom=10, max_depth=2, dry_run=True)
    for c in cells:
        assert c.zoom_level == 10 + c.depth * 2


async def test_dry_run_start_zoom_12(registry):
    """start_zoom=12: chain is 12 -> 14 -> 16."""
    cells = await analyze_cell(MARCO, registry, start_zoom=12, max_depth=2, dry_run=True)
    assert len(cells) == 273
    for c in cells:
        assert c.zoom_level == 12 + c.depth * 2


async def test_dry_run_zoom_12_depth_3_terminal(registry):
    """start_zoom=12, max_depth=3: chain 12->14->16. Zoom 16 is terminal now (zoom 18 is owned by the structure phase, not the recursion)."""
    cells = await analyze_cell(MARCO, registry, start_zoom=12, max_depth=3, dry_run=True)
    zoom_18_cells = [c for c in cells if c.zoom_level == 18]
    assert len(zoom_18_cells) == 0
    zoom_16_cells = [c for c in cells if c.zoom_level == 16]
    assert len(zoom_16_cells) > 0
    for c in zoom_16_cells:
        assert c.children_ids == []


async def test_dry_run_size_miles(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    for c in cells:
        expected = ground_coverage_miles(c.zoom_level, c.center[0])
        assert abs(c.size_miles - expected) < 0.01


async def test_dry_run_all_cells_have_analysis(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    for c in cells:
        assert c.analysis is not None
        assert c.analysis.model_used == "placeholder"
        assert len(c.analysis.sub_scores) == 16


async def test_dry_run_sub_cell_centers_within_parent(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    for c in cells:
        for sc in c.analysis.sub_scores:
            lat, lon = sc.center
            assert c.bbox.south < lat < c.bbox.north
            assert c.bbox.west < lon < c.bbox.east


async def test_dry_run_metadata_saved(registry, tmp_path):
    cells = await analyze_cell(
        MARCO, registry, max_depth=1, dry_run=True, output_dir=str(tmp_path),
    )
    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert len(meta) == len(cells)

    root_meta = meta[0]
    assert root_meta["cell_id"] == "root"
    assert root_meta["depth"] == 0
    assert root_meta["zoom"] == 10
    assert root_meta["parent_id"] is None
    assert "bbox" in root_meta
    assert root_meta["providers"] == []


# --- Depth limiting ---


async def test_max_depth_0(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=0, dry_run=True)
    assert len(cells) == 1
    assert cells[0].children_ids == []


async def test_max_depth_1(registry):
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    assert len(cells) == 17


# --- Threshold filtering ---


async def test_high_threshold_prunes_all(registry):
    cells = await analyze_cell(MARCO, registry, threshold=6.0, max_depth=2, dry_run=True)
    assert len(cells) == 1


# --- Start zoom validation ---


async def test_invalid_start_zoom_raises(registry):
    for bad_zoom in [9, 11, 13, 15]:
        with pytest.raises(ValueError, match="start_zoom"):
            await analyze_cell(MARCO, registry, start_zoom=bad_zoom, dry_run=True)


# --- Hard cap on API calls ---


async def test_hard_cap_stops_at_limit(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=5, max_depth=2, output_dir=str(tmp_path),
    )
    assert len(cells) == 5


async def test_hard_cap_all_cells_have_images(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=5, max_depth=2, output_dir=str(tmp_path),
    )
    for c in cells:
        assert c.image_path is not None
        assert len(c.provider_images) > 0


async def test_hard_cap_tree_structure(fast_registry, tmp_path):
    """With 5 calls: root, root-1, root-1-1, root-1-2, root-1-3."""
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=5, max_depth=2, output_dir=str(tmp_path),
    )
    ids = {c.id for c in cells}
    assert "root" in ids
    assert "root-1" in ids
    assert "root-1-1" in ids
    assert "root-1-2" in ids
    assert "root-1-3" in ids


async def test_hard_cap_1_returns_root_only(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=1, max_depth=2, output_dir=str(tmp_path),
    )
    assert len(cells) == 1
    assert cells[0].id == "root"
    assert cells[0].children_ids == []


async def test_hard_cap_metadata(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=5, max_depth=2, output_dir=str(tmp_path),
    )
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert len(meta) == len(cells) == 5


# --- Image files ---


async def test_images_written_to_disk(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=10, max_depth=1, output_dir=str(tmp_path),
    )
    for c in cells:
        assert Path(c.image_path).exists()


async def test_image_filenames_follow_convention(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=17, max_depth=1, output_dir=str(tmp_path),
    )
    root = cells[0]
    assert Path(root.image_path).name == "z0.png"
    for c in cells:
        if c.depth == 1:
            assert Path(c.image_path).name.startswith("z0_")


# --- Multi-provider at structure level ---


def _vision_patches():
    """Context manager that mocks all vision calls + sleep for non-dry-run tests."""
    return (
        patch("readwater.pipeline.cell_analyzer.asyncio.sleep", new_callable=AsyncMock),
        patch("readwater.pipeline.cell_analyzer.analyze_grid_image", new_callable=AsyncMock, return_value=_placeholder_grid_result()),
        patch("readwater.pipeline.cell_analyzer.run_structure_phase", new_callable=AsyncMock, return_value=_placeholder_structure_phase_result()),
        patch("readwater.pipeline.cell_analyzer.confirm_fishing_water", new_callable=AsyncMock, return_value=_placeholder_confirm_result()),
        patch("readwater.pipeline.cell_analyzer.draw_grid_overlay", side_effect=lambda p, **kw: p),
    )


async def test_structure_zoom_multi_provider(tmp_path):
    """Two providers for 'structure' role — depth-2 cells at zoom 16 get 2 images."""
    p1 = PlaceholderProvider(provider_name="google_static")
    p2 = PlaceholderProvider(provider_name="naip")
    reg = ImageProviderRegistry()
    reg.register(p1, ["overview", "structure"])
    reg.register(p2, ["structure"])

    with _vision_patches()[0], _vision_patches()[1], _vision_patches()[2], _vision_patches()[3], _vision_patches()[4]:
        cells = await analyze_cell(
            MARCO, reg, start_zoom=12, max_depth=2, max_api_calls=21,
            output_dir=str(tmp_path),
        )

    structure_cells = [c for c in cells if c.zoom_level == 16]
    assert len(structure_cells) > 0
    full = [c for c in structure_cells if len(c.provider_images) == 2]
    assert len(full) > 0
    for c in full:
        assert "google_static" in c.provider_images
        assert "naip" in c.provider_images


async def test_structure_filenames_have_provider_suffix(tmp_path):
    p1 = PlaceholderProvider(provider_name="google_static")
    p2 = PlaceholderProvider(provider_name="naip")
    reg = ImageProviderRegistry()
    reg.register(p1, ["overview", "structure"])
    reg.register(p2, ["structure"])

    with _vision_patches()[0], _vision_patches()[1], _vision_patches()[2], _vision_patches()[3], _vision_patches()[4]:
        cells = await analyze_cell(
            MARCO, reg, start_zoom=12, max_depth=2, max_api_calls=21,
            output_dir=str(tmp_path),
        )

    structure_cells = [c for c in cells if c.zoom_level == 16 and len(c.provider_images) == 2]
    assert len(structure_cells) > 0
    c = structure_cells[0]
    filenames = [Path(p).name for p in c.provider_images.values()]
    assert any("google_static" in f for f in filenames)
    assert any("naip" in f for f in filenames)


async def test_structure_api_calls_count_per_provider(tmp_path):
    """2 providers at structure level means 2 API calls per cell + 1 zoom-15 context fetch."""
    p1 = PlaceholderProvider(provider_name="a")
    p2 = PlaceholderProvider(provider_name="b")
    reg = ImageProviderRegistry()
    reg.register(p1, ["overview", "structure"])
    reg.register(p2, ["structure"])

    # Budget: root(1) + 1 depth-1(1) + 2 depth-2 cells at (2 providers + 1 z15 ctx) = 1 + 1 + 2*3 = 8.
    with _vision_patches()[0], _vision_patches()[1], _vision_patches()[2], _vision_patches()[3], _vision_patches()[4]:
        cells = await analyze_cell(
            MARCO, reg, start_zoom=12, max_depth=2, max_api_calls=8,
            output_dir=str(tmp_path),
        )

    structure_cells = [c for c in cells if c.zoom_level == 16 and len(c.provider_images) == 2]
    assert len(structure_cells) == 2


async def test_overview_single_provider_no_suffix(fast_registry, tmp_path):
    """Overview role uses default provider, no provider suffix in filename."""
    cells = await analyze_cell(
        MARCO, fast_registry, start_zoom=10, max_depth=0, output_dir=str(tmp_path),
    )
    root = cells[0]
    assert Path(root.image_path).name == "z0.png"
    assert len(root.provider_images) == 1


# --- Metadata content ---


async def test_metadata_records_provider(fast_registry, tmp_path):
    cells = await analyze_cell(
        MARCO, fast_registry, max_api_calls=1, max_depth=0, output_dir=str(tmp_path),
    )
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta[0]["providers"] == ["placeholder"]
    assert "placeholder" in meta[0]["provider_images"]


# --- Tiling at pipeline level ---


async def test_sub_cells_tile_parent_at_depth_1(registry):
    """16 depth-1 cells should tile the root bbox with no gaps."""
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    root = cells[0]
    children = [c for c in cells if c.depth == 1]
    assert len(children) == 16

    all_norths = [c.bbox.north for c in children]
    all_souths = [c.bbox.south for c in children]
    all_easts = [c.bbox.east for c in children]
    all_wests = [c.bbox.west for c in children]

    assert abs(max(all_norths) - root.bbox.north) < 1e-6
    assert abs(min(all_souths) - root.bbox.south) < 1e-6
    assert abs(max(all_easts) - root.bbox.east) < 1e-6
    assert abs(min(all_wests) - root.bbox.west) < 1e-6


async def test_sub_cell_half_dimensions(registry):
    """Each depth-1 cell should be 1/4 of root's width and height in degrees."""
    cells = await analyze_cell(MARCO, registry, max_depth=1, dry_run=True)
    root = cells[0]
    root_h = root.bbox.north - root.bbox.south
    root_w = root.bbox.east - root.bbox.west

    for c in cells:
        if c.depth == 1:
            h = c.bbox.north - c.bbox.south
            w = c.bbox.east - c.bbox.west
            assert abs(h - root_h / 4) < 1e-6
            assert abs(w - root_w / 4) < 1e-6


# --- Mocked Claude vision: selective pruning ---


def _mock_grid_result(high_cells=(1, 2, 3, 4)):
    """Grid analysis result: high scores for specified cells, low for others."""
    return {
        "summary": "Mock analysis",
        "sub_scores": [
            {
                "cell_number": i,
                "score": 8.0 if i in high_cells else 1.0,
                "reasoning": f"Cell {i} mock",
            }
            for i in range(1, 17)
        ],
        "hydrology_notes": "Mock hydrology",
        "raw_response": "",
    }


async def test_vision_pruning_only_high_cells_recurse(tmp_path):
    """Mock Claude to score cells 1-4 high (8.0) and 5-16 low (1.0).
    With threshold=4.0, only cells 1-4 should recurse into children.
    Cells 1-4 map to row 0 (r0c0, r0c1, r0c2, r0c3)."""
    reg = ImageProviderRegistry()
    reg.register(PlaceholderProvider(), ["overview", "structure"])

    mock_result = _mock_grid_result(high_cells=(1, 2, 3, 4))

    with (
        patch("readwater.pipeline.cell_analyzer.asyncio.sleep", new_callable=AsyncMock),
        patch("readwater.pipeline.cell_analyzer.analyze_grid_image", new_callable=AsyncMock, return_value=mock_result),
        patch("readwater.pipeline.cell_analyzer.confirm_fishing_water", new_callable=AsyncMock, return_value=_placeholder_confirm_result()),
        patch("readwater.pipeline.cell_analyzer.draw_grid_overlay", side_effect=lambda p, **kw: p),
    ):
        cells = await analyze_cell(
            MARCO, reg, start_zoom=10, max_depth=1,
            max_api_calls=100, output_dir=str(tmp_path),
        )

    root = cells[0]
    # Root should have exactly 4 children (cells 1-4 = row 0)
    assert len(root.children_ids) == 4
    assert root.children_ids == ["root-1", "root-2", "root-3", "root-4"]

    # Total cells: root + 4 children = 5
    assert len(cells) == 5

    # Children should all be row 0
    for c in cells[1:]:
        assert c.depth == 1
        assert c.parent_id == "root"
