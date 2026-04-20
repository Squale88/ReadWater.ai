"""Central registry of test cells used by every validation script.

Having one authoritative CELLS dict means mask-generation and
prompt-experiment scripts can't drift on center coordinates, zoom, or
parent-context strings. Each entry has everything downstream code needs:

  - cell_center:     (lat, lon) of the zoom-16 tile center
  - parent:          parent zoom-14 cell id (used for image filenames)
  - cell_num:        1-16 cell number within parent (row-major, 1-indexed)
  - zoom:            the zoom level represented by the tile (16 for all v1)
  - parent_context:  prose description of the parent cell, injected into
                     the discovery prompt
  - z16_image:       path to the zoom-16 tile PNG (pre-fetched)
  - z15_image:       path to the zoom-15 parent-context tile PNG (pre-fetched)

Centers are derived from the row/col of the cell inside its parent bbox
using the exact math `readwater.pipeline.cell_analyzer._subdivide_bbox`
uses (confirmed by round-tripping root-10-8 and root-11-5 against the
existing hand-copied values in prompt_experiment.py).

The 8 cells below span the Rookery Bay / Marco Island / Naples area from
roughly N 26.10 to N 25.97 and W -81.80 to W -81.71 — enough diversity
of structure types (open bay, mangrove shoreline, tidal cuts, oyster bars,
grass flats) to evaluate whether the evidence-injection approach holds up
outside the original 2-cell test set.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
IMAGES_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"


# Parent zoom-14 cell bboxes (copied verbatim from
# data/areas/rookery_bay_v2/images/metadata.json). Used to compute each
# sub-cell center so no value is hand-typed twice.
_PARENT_BBOX = {
    "root-2":  dict(north=26.128688889316162, south=26.07920794465808,
                    east=-81.746663,          west=-81.80172953716598),
    "root-6":  dict(north=26.07920794465808,  south=26.029727,
                    east=-81.746663,          west=-81.80172953716598),
    "root-7":  dict(north=26.07920794465808,  south=26.029727,
                    east=-81.69159646283401,  west=-81.746663),
    "root-10": dict(north=26.029727,          south=25.980246055341922,
                    east=-81.746663,          west=-81.80172953716598),
    "root-11": dict(north=26.029727,          south=25.980246055341922,
                    east=-81.69159646283401,  west=-81.746663),
    "root-15": dict(north=25.980246055341922, south=25.93076511068384,
                    east=-81.69159646283401,  west=-81.746663),
}

_SECTIONS = 4  # pipeline uses a 4x4 sub-cell grid


def _sub_cell_center(parent: str, cell_num: int) -> tuple[float, float]:
    """Return (lat, lon) of the given 1-indexed sub-cell inside its parent."""
    bb = _PARENT_BBOX[parent]
    row = (cell_num - 1) // _SECTIONS
    col = (cell_num - 1) % _SECTIONS
    cell_h = (bb["north"] - bb["south"]) / _SECTIONS
    cell_w = (bb["east"] - bb["west"]) / _SECTIONS
    lat = bb["north"] - (row + 0.5) * cell_h
    lon = bb["west"] + (col + 0.5) * cell_w
    return (round(lat, 6), round(lon, 6))


def _image_paths(parent: str, cell_num: int) -> tuple[Path, Path]:
    """Return (z16_path, z15_path) for a given sub-cell.

    Matches the naming convention produced by cell_analyzer._image_filename,
    e.g. cell ``root-10-8`` -> ``z0_10_8.png`` and the z15 context is
    ``z0_10_8_context_z15.png``.
    """
    parent_num = parent.removeprefix("root-")
    stem = f"z0_{parent_num}_{cell_num}"
    return (
        IMAGES_ROOT / f"{stem}.png",
        IMAGES_ROOT / f"{stem}_context_z15.png",
    )


# Parent-context strings by parent cell. These are short prose descriptions
# of the zoom-14 area each sub-cell falls inside of, used to prime the
# discovery prompt. They don't affect evidence signals — only Claude's
# framing of the tile.
_PARENT_CONTEXTS = {
    "root-2": (
        "Rookery Bay NERR, SW Florida. Parent zoom-14 cell covers the "
        "northwestern edge of the reserve: mangrove-lined coves, narrow "
        "tidal creeks, and shallow estuarine flats connecting to larger "
        "bays to the south."
    ),
    "root-6": (
        "Rookery Bay NERR, SW Florida. Parent zoom-14 cell covers interior "
        "estuarine waters on the west side of the reserve, with mangrove "
        "islands, tidal cuts, and oyster-rich shoreline edges."
    ),
    "root-7": (
        "Rookery Bay / Marco Island, SW Florida. Parent zoom-14 cell covers "
        "a mid-bay stretch with charted boating channels, mangrove points, "
        "and grass-flat basins."
    ),
    "root-10": (
        "Rookery Bay, SW Florida. Parent zoom-14 cell shows mangrove-lined "
        "estuarine shoreline with tidal cuts and shallow basins."
    ),
    "root-11": (
        "Rookery Bay, SW Florida. Parent zoom-14 cell shows interior bay "
        "water with mangrove islands and connecting channels."
    ),
    "root-15": (
        "Marco Island / south Rookery Bay, SW Florida. Parent zoom-14 cell "
        "covers southern bay water with charted boating passes, mangrove "
        "edges, and oyster-rich shallow shoals."
    ),
}


def _build_cell(parent: str, cell_num: int) -> dict:
    cell_id = f"{parent}-{cell_num}"
    center = _sub_cell_center(parent, cell_num)
    z16, z15 = _image_paths(parent, cell_num)
    return {
        "cell_id": cell_id,
        "cell_center": center,
        "center": center,  # alias — mask scripts read "center"
        "zoom": 16,
        "parent": parent,
        "cell_num": cell_num,
        "parent_context": _PARENT_CONTEXTS[parent],
        "z16_image": z16,
        "z15_image": z15,
        # Legacy key names kept for prompt_experiment.py compatibility.
        "z16": z16,
        "z15": z15,
    }


# --- The public CELLS dict ---
#
# 8 cells total:
#   - 2 originals kept for A/B comparability with prior evidence runs
#   - 6 new cells added for broader validation before committing to the
#     next architectural layer (candidate-feature generation).
_CELL_ORDER = [
    ("root-10", 8),    # original — mangrove drain + creek mouths
    ("root-11", 5),    # original — open basin, few organizing features
    ("root-6",  10),   # new — interior bay, mangrove island cluster
    ("root-2",  9),    # new — NW coves with narrow tidal cuts
    ("root-7",  14),   # new — charted channel + grass flats
    ("root-10", 3),    # new — north edge of root-10, creek system
    ("root-11", 1),    # new — NW corner of root-11, point/cove mix
    ("root-15", 3),    # new — south bay, charted pass + oyster shoals
]

CELLS: dict[str, dict] = {
    f"{parent}-{num}": _build_cell(parent, num) for parent, num in _CELL_ORDER
}


def list_ids() -> list[str]:
    """Return cell ids in the canonical test-order (originals first, then new)."""
    return list(CELLS.keys())


if __name__ == "__main__":
    # Quick sanity-check: print the registry so you can compare against
    # metadata.json or a map before running the mask pipelines.
    import json

    for cid, spec in CELLS.items():
        print(f"{cid:12s}  center={spec['cell_center']}  "
              f"z16={spec['z16_image'].name}  z15_exists={spec['z15_image'].exists()}  "
              f"z16_exists={spec['z16_image'].exists()}")
    print()
    print(json.dumps(
        {cid: {"cell_center": spec["cell_center"], "parent": spec["parent"]}
         for cid, spec in CELLS.items()},
        indent=2,
    ))
