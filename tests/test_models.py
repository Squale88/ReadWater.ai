"""Tests for data models."""

from readwater.models.cell import BoundingBox, Cell
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
