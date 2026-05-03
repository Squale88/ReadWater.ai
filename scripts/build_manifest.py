"""Scan an area's existing on-disk artifacts and (re)generate manifest.json.

The manifest is the canonical artifact index for an area — see
``src/readwater/areas/__init__.py`` for the schema. This script is the
one-time bootstrap that turns the current "files scattered across sibling
dirs, latest-by-glob" world into the new "everything indexed in
manifest.json" world.

It's also safe to re-run after the pipeline has produced new outputs —
the script discovers what's on disk and overwrites the manifest. The
pipeline itself will eventually update the manifest in place after each
write, but this scanner is the fallback / catchup tool.

Usage:
  python scripts/build_manifest.py --area rookery_bay_v2
  python scripts/build_manifest.py --area rookery_bay_v2 --dry-run

For each cell, records (when present on disk):
  - z16_image, z15_context        (cell sat tiles, from discovery)
  - water_mask, z14_wide_styled   (water mask + the wide context tile)
  - seagrass_mask, oyster_mask    (FWC habitat masks)
  - anchors, anchors_overlay      (latest cv_all_<ts>.{json,png})
  - anchors_schema_version        (read from cv_all JSON's "phase" field)
  - detector_drains/islands/points/pockets (latest cv_<kind>_<ts>.json)

Cells are discovered from on-disk z16 sat tiles named ``z0_<p>_<c>.png``
in the area's images dir. We do NOT use ``scripts/_cells.py`` (the legacy
9-cell hand-picked test fixture) — that's being deprecated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make src/ importable when running directly via ``python scripts/...``
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from readwater import storage  # noqa: E402
from readwater.areas import MANIFEST_SCHEMA_VERSION  # noqa: E402

Z16_PATTERN = re.compile(r"^z0_(\d+)_(\d+)\.png$")


def _discover_cell_ids(area_id: str) -> list[str]:
    """Return all z16 cell ids in the area, by scanning the images dir.

    We deliberately skip variants like ``..._context_z15.png`` and
    ``..._grid.png`` — only the bare ``z0_<p>_<c>.png`` form is treated
    as a real z16 cell.
    """
    images_dir = storage.area_root(area_id) / "images"
    if not images_dir.exists():
        return []
    ids: set[str] = set()
    for p in images_dir.glob("z0_*.png"):
        m = Z16_PATTERN.match(p.name)
        if m:
            ids.add(f"root-{m.group(1)}-{m.group(2)}")
    return sorted(ids)


def _latest_cv_json(structures_dir: Path, prefix: str) -> Path | None:
    """Return the newest ``<prefix>*.json`` file or None."""
    if not structures_dir.exists():
        return None
    matches = sorted(structures_dir.glob(f"{prefix}*.json"))
    return matches[-1] if matches else None


def _matching_overlay(json_path: Path) -> Path | None:
    """Given a cv_*.json path, return the sibling .png if it exists."""
    candidate = json_path.with_suffix(".png")
    return candidate if candidate.exists() else None


def _maybe_record(entry: dict, key: str, path: Path) -> None:
    if path.exists():
        entry[key] = storage.relative_to_data_root(path)


def _build_cell_entry(area_id: str, cell_id: str) -> dict:
    """Inspect on-disk artifacts for one cell and produce its manifest entry."""
    entry: dict = {}

    # Discovery outputs
    _maybe_record(entry, "z16_image", storage.z16_image_path(area_id, cell_id))
    _maybe_record(entry, "z15_context", storage.z15_context_path(area_id, cell_id))

    # Water mask + context tile (transitional sibling-dir layout)
    _maybe_record(entry, "water_mask", storage.water_mask_path(area_id, cell_id))
    _maybe_record(entry, "z14_wide_styled",
                  storage.water_z14_wide_styled_path(area_id, cell_id))

    # Habitat masks
    _maybe_record(entry, "seagrass_mask",
                  storage.seagrass_mask_path(area_id, cell_id))
    _maybe_record(entry, "oyster_mask",
                  storage.oyster_mask_path(area_id, cell_id))

    # Per-detector latest JSONs
    structures_dir = storage.cell_structures_dir(area_id, cell_id)
    for kind, prefix in [
        ("drains", "cv_drains_"),
        ("islands", "cv_islands_"),
        ("points", "cv_points_"),
        ("pockets", "cv_pockets_"),
    ]:
        latest = _latest_cv_json(structures_dir, prefix)
        if latest is not None:
            entry[f"detector_{kind}"] = storage.relative_to_data_root(latest)

    # Orchestrator (anchors)
    anchors_json = _latest_cv_json(structures_dir, "cv_all_")
    if anchors_json is not None:
        entry["anchors"] = storage.relative_to_data_root(anchors_json)
        overlay = _matching_overlay(anchors_json)
        if overlay is not None:
            entry["anchors_overlay"] = storage.relative_to_data_root(overlay)
        # Pull schema_version (recorded as "phase" in the cv_all JSON).
        try:
            payload = json.loads(anchors_json.read_text(encoding="utf-8"))
            phase = payload.get("phase")
            if phase is not None:
                entry["anchors_schema_version"] = str(phase)
        except (OSError, json.JSONDecodeError):
            pass

    return entry


def build_manifest(area_id: str) -> dict:
    cell_ids = _discover_cell_ids(area_id)
    cells: dict[str, dict] = {}
    for cid in cell_ids:
        cells[cid] = _build_cell_entry(area_id, cid)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "area_id": area_id,
        "cells": cells,
    }


def _summarize(manifest: dict) -> str:
    cells = manifest.get("cells", {})
    n = len(cells)
    has = lambda key: sum(1 for c in cells.values() if key in c)  # noqa: E731
    parts = [
        f"  cells:           {n}",
        f"  z16_image:       {has('z16_image'):>3} / {n}",
        f"  z15_context:     {has('z15_context'):>3} / {n}",
        f"  water_mask:      {has('water_mask'):>3} / {n}",
        f"  z14_wide_styled: {has('z14_wide_styled'):>3} / {n}",
        f"  seagrass_mask:   {has('seagrass_mask'):>3} / {n}",
        f"  oyster_mask:     {has('oyster_mask'):>3} / {n}",
        f"  detector_drains: {has('detector_drains'):>3} / {n}",
        f"  detector_islands:{has('detector_islands'):>3} / {n}",
        f"  detector_points: {has('detector_points'):>3} / {n}",
        f"  detector_pockets:{has('detector_pockets'):>3} / {n}",
        f"  anchors:         {has('anchors'):>3} / {n}",
    ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True,
                        help="Area id, e.g. rookery_bay_v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary but don't write manifest.json")
    args = parser.parse_args()

    area_root = storage.area_root(args.area)
    if not area_root.exists():
        print(f"Area dir does not exist: {area_root}", file=sys.stderr)
        return 1

    print(f"Scanning {area_root} ...")
    manifest = build_manifest(args.area)
    print(_summarize(manifest))

    out_path = storage.area_manifest_path(args.area)
    if args.dry_run:
        print(f"\n[dry-run] would write {out_path}")
        return 0

    # Use atomic write; preserve any extra top-level fields the user added
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for k, v in existing.items():
                if k not in {"schema_version", "area_id", "generated_at",
                             "cells"}:
                    manifest[k] = v
        except json.JSONDecodeError:
            pass

    from datetime import datetime, timezone
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    storage.atomic_write_json(out_path, manifest)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
