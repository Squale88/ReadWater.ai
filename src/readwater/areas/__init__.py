"""Cell-and-area registry — the canonical "what cells exist and what
artifacts do they have" API.

The trip planner, CV pipeline, downstream LLM agents, and any cloud-side
caller all read through ``Area("rookery_bay_v2")``. The class loads the
area's ``manifest.json`` once and exposes typed accessors for every
artifact, so callers never glob the filesystem and never construct paths
by hand.

The manifest is the source-of-truth. It's produced by
``scripts/build_manifest.py`` (one-time scanner over existing on-disk
artifacts) and updated by the CV pipeline as it produces new outputs.
Cloud-side callers will fetch ``manifest.json`` once and then have
addressable URLs for every artifact in the area — no LIST operations,
no globs.

Manifest schema (v1):

  {
    "schema_version": "1.0",
    "area_id": "rookery_bay_v2",
    "generated_at": "2026-05-03T...",
    "cells": {
      "<cell_id>": {
        "z16_image":          "areas/<area>/images/z0_<p>_<c>.png",
        "z15_context":        "areas/<area>/images/z0_<p>_<c>_context_z15.png",
        "water_mask":         "areas/<area>_google_water/<cell>_water_mask.png",
        "z14_wide_styled":    "areas/<area>_google_water/<cell>_wide_z14_styled.png",
        "seagrass_mask":      "areas/<area>_habitats/<cell>_seagrass_mask.png",
        "oyster_mask":        "areas/<area>_habitats/<cell>_oyster_mask.png",
        "anchors":            "areas/<area>/images/structures/<cell>/cv_all_<ts>.json",
        "anchors_overlay":    "areas/<area>/images/structures/<cell>/cv_all_<ts>.png",
        "anchors_schema_version": "3b",
        "detector_drains":    "areas/<area>/images/structures/<cell>/cv_drains_<ts>.json",
        "detector_islands":   "...",
        "detector_points":    "...",
        "detector_pockets":   "..."
      },
      ...
    }
  }

Every value is a path string relative to ``data_root()`` (see
``readwater.storage``). Missing artifacts are simply omitted from the
cell's entry — callers check ``cell.has(kind)`` or get ``None`` from a
typed accessor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from readwater import storage

MANIFEST_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


class Cell:
    """One z16 cell within an area, plus typed accessors for its artifacts.

    Construct via ``area.cell(cell_id)`` — never directly.
    """

    def __init__(self, area: "Area", cell_id: str, entry: dict):
        self.area = area
        self.cell_id = cell_id
        self._entry = entry

    # ---- Identity ----

    def __repr__(self) -> str:
        return f"Cell(area={self.area.area_id!r}, cell={self.cell_id!r})"

    @property
    def parent_num(self) -> str:
        return self.cell_id.removeprefix("root-").split("-", 1)[0]

    @property
    def child_num(self) -> str:
        return self.cell_id.removeprefix("root-").split("-", 1)[1]

    @property
    def parent_id(self) -> str:
        """Parent z14 cell id, e.g. ``"root-10"``."""
        return self._entry.get("parent", f"root-{self.parent_num}")

    @property
    def cell_num(self) -> int:
        """1-indexed cell number within the parent (1..16 for a 4x4 grid)."""
        return int(self._entry.get("cell_num", self.child_num))

    @property
    def center(self) -> tuple[float, float] | None:
        """(lat, lon) of the cell center, or None if not in the manifest."""
        c = self._entry.get("center")
        if c is None:
            return None
        return (float(c[0]), float(c[1]))

    @property
    def parent_bbox(self) -> dict | None:
        """Parent z14 bbox dict (north/south/east/west) or None."""
        return self.area.parent_bbox(self.parent_id)

    # ---- Generic accessors ----

    def has(self, kind: str) -> bool:
        """True iff the manifest records an artifact of this kind."""
        rel = self._entry.get(kind)
        if not rel:
            return False
        return storage.absolute_from_data_root(rel).exists()

    def path(self, kind: str) -> Path | None:
        """Absolute path for an artifact of this kind, or None if absent."""
        rel = self._entry.get(kind)
        return storage.absolute_from_data_root(rel) if rel else None

    # ---- Typed accessors for the well-known kinds ----

    @property
    def z16_image(self) -> Path | None:
        return self.path("z16_image")

    @property
    def z15_context(self) -> Path | None:
        return self.path("z15_context")

    @property
    def water_mask(self) -> Path | None:
        return self.path("water_mask")

    @property
    def z14_wide_styled(self) -> Path | None:
        return self.path("z14_wide_styled")

    @property
    def seagrass_mask(self) -> Path | None:
        return self.path("seagrass_mask")

    @property
    def oyster_mask(self) -> Path | None:
        return self.path("oyster_mask")

    @property
    def anchors_json(self) -> Path | None:
        return self.path("anchors")

    @property
    def anchors_overlay(self) -> Path | None:
        return self.path("anchors_overlay")

    @property
    def anchors_schema_version(self) -> str | None:
        return self._entry.get("anchors_schema_version")

    def detector_json(self, kind: str) -> Path | None:
        """Latest detector JSON for ``kind`` (one of "drains", "islands",
        "points", "pockets"), or None.
        """
        return self.path(f"detector_{kind}")

    # ---- Mutation (for pipeline writers) ----

    def set_artifact(self, kind: str, path: Path,
                     extra: dict | None = None) -> None:
        """Record (or replace) an artifact's manifest entry. Caller must
        invoke ``area.save_manifest()`` afterward to persist.
        """
        self._entry[kind] = storage.relative_to_data_root(path)
        if extra:
            for k, v in extra.items():
                self._entry[k] = v

    def remove_artifact(self, kind: str) -> None:
        self._entry.pop(kind, None)

    @property
    def manifest_entry(self) -> dict:
        """The raw manifest dict for this cell (for advanced callers)."""
        return self._entry


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


class Area:
    """A geographic area (e.g. ``"rookery_bay_v2"``) and all its cells.

    Loads ``manifest.json`` lazily on construction. Use ``cells()`` to
    iterate, or ``cell(cell_id)`` to look up by id.
    """

    def __init__(self, area_id: str):
        self.area_id = area_id
        self.path = storage.area_root(area_id)
        self._manifest_path = storage.area_manifest_path(area_id)
        self._manifest = self._load_manifest()
        self._cells = {
            cid: Cell(self, cid, entry)
            for cid, entry in self._manifest.get("cells", {}).items()
        }

    def _load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest at {self._manifest_path}. Run "
                f"scripts/build_manifest.py --area {self.area_id} first."
            )
        return json.loads(self._manifest_path.read_text(encoding="utf-8"))

    # ---- Read accessors ----

    def cell(self, cell_id: str) -> Cell:
        return self._cells[cell_id]

    def cells(self) -> Iterator[Cell]:
        """Iterate cells in id-sorted order."""
        for cid in sorted(self._cells):
            yield self._cells[cid]

    def cell_ids(self) -> list[str]:
        return sorted(self._cells)

    def has_cell(self, cell_id: str) -> bool:
        return cell_id in self._cells

    def parent_bbox(self, parent_id: str) -> dict | None:
        """Return the parent z14 bbox dict (or None) for ``parent_id`` like
        ``"root-10"``. Used by Cell.parent_bbox + by the water_mask pipeline.
        """
        parents = self._manifest.get("parents", {})
        entry = parents.get(parent_id)
        return entry.get("bbox") if entry else None

    @property
    def schema_version(self) -> str:
        return self._manifest.get("schema_version", "?")

    @property
    def generated_at(self) -> str | None:
        return self._manifest.get("generated_at")

    # ---- Mutation ----

    def add_cell(self, cell_id: str, entry: dict | None = None) -> Cell:
        """Create or replace a cell entry. Caller must call save_manifest()."""
        entry = entry or {}
        self._manifest.setdefault("cells", {})[cell_id] = entry
        cell = Cell(self, cell_id, entry)
        self._cells[cell_id] = cell
        return cell

    def save_manifest(self) -> None:
        """Atomically rewrite manifest.json reflecting current state."""
        self._manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
        self._manifest["area_id"] = self.area_id
        self._manifest["generated_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        # Re-serialize cells dict in id-sorted order so diffs are stable
        cells = {cid: self._cells[cid].manifest_entry for cid in sorted(self._cells)}
        self._manifest["cells"] = cells
        storage.atomic_write_json(self._manifest_path, self._manifest)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def open_area(area_id: str) -> Area:
    """Same as ``Area(area_id)`` — provided for consistency with future
    factory functions (e.g. cloud-backed areas).
    """
    return Area(area_id)
