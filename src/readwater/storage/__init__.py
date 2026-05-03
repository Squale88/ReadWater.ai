"""Cloud-friendly storage abstraction for area data.

Every path that points at on-disk data flows through this module. Hardcoded
paths (``Path("D:/dropbox_root/...")``) are forbidden in pipeline code —
they only exist here, behind functions that can be swapped out for a cloud
backend later (S3, GCS, etc.) without touching callers.

The split:

  * ``data_root()``           — root of all data, env-overridable
  * ``area_root(area_id)``    — per-area directory under data_root
  * ``cell_artifact_path(area_id, cell_id, kind, version=None)``
                              — canonical path for an individual artifact
  * ``atomic_write_text``     — write-then-rename, safe under concurrent
                                readers and resilient to mid-write crashes
  * ``atomic_write_bytes``    — same, for binary payloads
  * ``atomic_write_json``     — convenience over atomic_write_text

When we move to cloud, the functions are reimplemented to return URI-like
objects (or to perform staged uploads under the hood), and callers don't
need to change. That's the point: this module is the seam.

Path conventions (current, on-disk):

  data_root() = $READWATER_DATA_ROOT or ``<repo_root>/data``
  area_root("rookery_bay_v2") = data_root() / "areas" / "rookery_bay_v2"
  Per-cell anchor JSON:
      area_root() / "images" / "structures" / <cell_id> / "cv_all_<ts>.json"

Mask paths (consolidated under area_root() as of PR 2):

  Water mask:   area_root() / "masks" / "water"    / "<cell>_water_mask.png"
  Z14 wide:     area_root() / "masks" / "water"    / "<cell>_wide_z14_styled.png"
  Seagrass:     area_root() / "masks" / "seagrass" / "<cell>_seagrass_mask.png"
  Oyster:       area_root() / "masks" / "oyster"   / "<cell>_oyster_mask.png"

Shared/cached habitat geojson lives at area_root() / "masks" / and is
keyed by habitat kind (oyster_beds.geojson, seagrass.geojson).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Repo root is fixed by where this file lives in the source tree. Don't
# read REPO_ROOT from anywhere else — let everyone go through data_root().
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def data_root() -> Path:
    """Root of all area data on disk.

    Defaults to ``<repo>/data``. Overridable via the ``READWATER_DATA_ROOT``
    env var so the same code can run against a developer's local checkout,
    a CI scratch dir, or (eventually) a cloud-mirrored local cache.
    """
    override = os.environ.get("READWATER_DATA_ROOT")
    if override:
        return Path(override)
    return _REPO_ROOT / "data"


def area_root(area_id: str) -> Path:
    """Root directory for a single area's data."""
    return data_root() / "areas" / area_id


def area_manifest_path(area_id: str) -> Path:
    """Path to the area's manifest.json (canonical artifact index)."""
    return area_root(area_id) / "manifest.json"


# ---------------------------------------------------------------------------
# Mask paths (consolidated under area_root() as of PR 2)
# ---------------------------------------------------------------------------


def masks_root(area_id: str) -> Path:
    return area_root(area_id) / "masks"


def water_masks_dir(area_id: str) -> Path:
    return masks_root(area_id) / "water"


def seagrass_masks_dir(area_id: str) -> Path:
    return masks_root(area_id) / "seagrass"


def oyster_masks_dir(area_id: str) -> Path:
    return masks_root(area_id) / "oyster"


def water_mask_path(area_id: str, cell_id: str) -> Path:
    return water_masks_dir(area_id) / f"{cell_id}_water_mask.png"


def water_mask_overlay_path(area_id: str, cell_id: str) -> Path:
    return water_masks_dir(area_id) / f"{cell_id}_water_overlay.png"


def water_styled_z16_path(area_id: str, cell_id: str) -> Path:
    return water_masks_dir(area_id) / f"{cell_id}_styled.png"


def water_z14_wide_styled_path(area_id: str, cell_id: str) -> Path:
    return water_masks_dir(area_id) / f"{cell_id}_wide_z14_styled.png"


def water_z13_isolation_styled_path(area_id: str, cell_id: str) -> Path:
    return water_masks_dir(area_id) / f"{cell_id}_wide_z13_styled.png"


def seagrass_mask_path(area_id: str, cell_id: str) -> Path:
    return seagrass_masks_dir(area_id) / f"{cell_id}_seagrass_mask.png"


def oyster_mask_path(area_id: str, cell_id: str) -> Path:
    return oyster_masks_dir(area_id) / f"{cell_id}_oyster_mask.png"


def oyster_beds_geojson_path(area_id: str) -> Path:
    """Area-level cached oyster-reef polygon geojson (shared across cells)."""
    return masks_root(area_id) / "oyster_beds.geojson"


def seagrass_geojson_path(area_id: str) -> Path:
    """Area-level cached seagrass polygon geojson (shared across cells)."""
    return masks_root(area_id) / "seagrass.geojson"


# ---------------------------------------------------------------------------
# Per-cell sat tile + structures dir
# ---------------------------------------------------------------------------


def _z16_filename(cell_id: str) -> str:
    """Cell id like ``root-10-8`` -> filename like ``z0_10_8.png``."""
    parent_num, child_num = cell_id.removeprefix("root-").split("-")
    return f"z0_{parent_num}_{child_num}.png"


def z16_image_path(area_id: str, cell_id: str) -> Path:
    return area_root(area_id) / "images" / _z16_filename(cell_id)


def z15_context_path(area_id: str, cell_id: str) -> Path:
    fname = _z16_filename(cell_id).replace(".png", "_context_z15.png")
    return area_root(area_id) / "images" / fname


def cell_structures_dir(area_id: str, cell_id: str) -> Path:
    return area_root(area_id) / "images" / "structures" / cell_id


def cell_artifact_path(
    area_id: str,
    cell_id: str,
    kind: str,
    version: str | None = None,
    extension: str = "json",
) -> Path:
    """Canonical path for one CV artifact for one cell.

    ``kind`` is the detector / orchestrator name (e.g. ``"drains"``,
    ``"islands"``, ``"all"``). When ``version`` is omitted, the path
    points at the un-versioned filename (rare; mostly for things like
    summary indices); normally callers pass a timestamp.

    The naming convention is preserved from the existing scripts:
    ``cv_<kind>_<version>.{json,png}``.
    """
    suffix = f"_{version}" if version else ""
    fname = f"cv_{kind}{suffix}.{extension}"
    return cell_structures_dir(area_id, cell_id) / fname


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` atomically.

    Strategy: write to a tempfile in the same directory, fsync, then
    ``os.replace`` to the target. On POSIX this is atomic (single rename).
    On Windows ``os.replace`` is also atomic on NTFS.

    Concurrent readers see either the old file or the new file, never a
    partial write. If the process dies mid-write, the target is unchanged.
    The cloud equivalent will be "upload to staging key, then rename" or
    a finalize-on-completion put.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup; don't mask the original error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, content.encode(encoding))


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Atomic JSON dump. Default str fallback so Path/datetime serialize."""
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


# ---------------------------------------------------------------------------
# Helpers for path<->relative conversions used by the manifest
# ---------------------------------------------------------------------------


def relative_to_data_root(path: Path) -> str:
    """Return ``path`` relative to data_root() with forward slashes.

    Manifest entries store paths relative to data_root() (NOT relative to
    the area dir), so the same manifest schema works whether the artifact
    lives under the area's own subdirectory or in a sibling dir (the
    transitional layout for masks before PR 2). POSIX-style separators are
    used regardless of host OS so manifests are byte-identical across
    Windows / Linux / Mac.
    """
    return path.resolve().relative_to(data_root().resolve()).as_posix()


def absolute_from_data_root(relative: str) -> Path:
    """Inverse of relative_to_data_root: resolve a manifest-relative path."""
    return data_root() / relative
