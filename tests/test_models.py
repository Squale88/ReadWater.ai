"""Tests for data models."""

from readwater.models.area_knowledge import AreaKnowledge
from readwater.models.cell import BoundingBox, Cell, CellAnalysis, CellScore
from readwater.models.context import (
    CellContext,
    DirectObservation,
)


def test_bounding_box_center():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    lat, lon = bbox.center
    assert lat == 27.5
    assert lon == -82.5


def test_cell_creation():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    cell = Cell(
        id="root",
        center=(27.5, -82.5),
        size_miles=30.0,
        depth=0,
        zoom_level=10,
        bbox=bbox,
    )
    assert cell.parent_id is None
    assert cell.depth == 0
    assert cell.zoom_level == 10
    assert cell.children_ids == []
    assert cell.analysis is None
    assert cell.image_path is None
    assert cell.provider_images == {}


def test_cell_provider_images():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    cell = Cell(
        id="root",
        center=(27.5, -82.5),
        size_miles=30.0,
        depth=0,
        zoom_level=10,
        bbox=bbox,
        image_path="/tmp/z0_root_google.png",
        provider_images={
            "google_static": "/tmp/z0_root_google.png",
            "naip": "/tmp/z0_root_naip.png",
        },
    )
    assert len(cell.provider_images) == 2
    assert "google_static" in cell.provider_images
    assert "naip" in cell.provider_images


def test_area_knowledge_add_and_query():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    root = Cell(
        id="root", center=(27.5, -82.5), size_miles=30.0,
        depth=0, zoom_level=10, bbox=bbox,
    )
    child = Cell(
        id="root-1-2", parent_id="root",
        center=(27.6, -82.4), size_miles=10.0,
        depth=1, zoom_level=12, bbox=bbox,
    )
    root.children_ids.append("root-1-2")

    ak = AreaKnowledge(name="Tampa Bay", root_center=(27.5, -82.5), root_size_miles=30.0)
    ak.add_cell(root)
    ak.add_cell(child)

    assert ak.total_cells_analyzed == 2
    assert len(ak.get_children("root")) == 1
    assert len(ak.get_cells_at_level(10)) == 1
    assert len(ak.get_cells_at_level(12)) == 1
    assert len(ak.get_cells_at_depth(0)) == 1
    assert len(ak.get_cells_at_depth(1)) == 1
    assert ak.get_leaf_cells() == [child]


def test_cellanalysis_defaults_context_to_none():
    """Backward-compat: CellAnalysis constructs without a context field."""
    analysis = CellAnalysis(
        overall_summary="test",
        sub_scores=[CellScore(row=0, col=0, score=5.0, summary="x", center=(27.5, -82.5))],
    )
    assert analysis.context is None


def test_cellanalysis_accepts_populated_context():
    ctx = CellContext(
        cell_id="root-6-11",
        zoom=16,
        observations=[
            DirectObservation(
                observation_id="root-6-11:obs:0",
                label="mangrove_shoreline",
            ),
        ],
    )
    analysis = CellAnalysis(
        overall_summary="test",
        sub_scores=[CellScore(row=0, col=0, score=5.0, summary="x", center=(27.5, -82.5))],
        context=ctx,
    )
    assert analysis.context is ctx
    assert analysis.context.observations[0].label == "mangrove_shoreline"


def test_cellanalysis_with_context_json_roundtrip():
    ctx = CellContext(
        cell_id="root-6-11",
        zoom=16,
        observations=[
            DirectObservation(
                observation_id="root-6-11:obs:0",
                label="mangrove_shoreline",
                confidence=0.8,
            ),
        ],
    )
    analysis = CellAnalysis(
        overall_summary="test",
        sub_scores=[CellScore(row=0, col=0, score=5.0, summary="x", center=(27.5, -82.5))],
        context=ctx,
    )
    dumped = analysis.model_dump_json()
    restored = CellAnalysis.model_validate_json(dumped)
    assert restored.context is not None
    assert restored.context.cell_id == "root-6-11"
    assert restored.context.observations[0].label == "mangrove_shoreline"


def test_cell_with_analysis_containing_context_roundtrip():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    ctx = CellContext(cell_id="root", zoom=12)
    analysis = CellAnalysis(
        overall_summary="root", sub_scores=[], context=ctx,
    )
    cell = Cell(
        id="root", center=(27.5, -82.5), size_miles=30.0,
        depth=0, zoom_level=12, bbox=bbox, analysis=analysis,
    )
    dumped = cell.model_dump_json()
    restored = Cell.model_validate_json(dumped)
    assert restored.analysis is not None
    assert restored.analysis.context is not None
    assert restored.analysis.context.cell_id == "root"
