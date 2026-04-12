"""Tests for data models."""

from readwater.models.cell import BoundingBox, Cell, CellAnalysis, CellScore, ZoomLevel
from readwater.models.area_knowledge import AreaKnowledge


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
        zoom_level=0,
        bbox=bbox,
    )
    assert cell.parent_id is None
    assert cell.zoom_level == 0
    assert cell.children_ids == []
    assert cell.analysis is None


def test_area_knowledge_add_and_query():
    bbox = BoundingBox(north=28.0, south=27.0, east=-82.0, west=-83.0)
    root = Cell(id="root", center=(27.5, -82.5), size_miles=30.0, zoom_level=0, bbox=bbox)
    child = Cell(
        id="root-1-2",
        parent_id="root",
        center=(27.6, -82.4),
        size_miles=10.0,
        zoom_level=1,
        bbox=bbox,
    )
    root.children_ids.append("root-1-2")

    ak = AreaKnowledge(name="Tampa Bay", root_center=(27.5, -82.5), root_size_miles=30.0)
    ak.add_cell(root)
    ak.add_cell(child)

    assert ak.total_cells_analyzed == 2
    assert len(ak.get_children("root")) == 1
    assert len(ak.get_cells_at_level(0)) == 1
    assert len(ak.get_cells_at_level(1)) == 1
    assert ak.get_leaf_cells() == [child]


def test_zoom_level_enum():
    assert ZoomLevel.MACRO == 0
    assert ZoomLevel.STRUCTURE == 4
