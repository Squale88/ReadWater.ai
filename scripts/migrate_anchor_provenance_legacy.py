"""One-shot migration: backfill Provenance on legacy cached AnchorStructure JSON.

Per `docs/PHASE_C_TASKS.md` TASK-4 + addendum: AnchorStructure now requires
a `Provenance`. Cached anchor JSON written before the schema change loads
fine under `extra='allow'` for unknown keys, but Pydantic refuses to
*validate* an old payload (e.g. when calling `AnchorStructure.model_validate`)
because `provenance` is missing.

Rather than silently default to an empty stub, this script writes a
honest legacy marker so Phase E and any future provenance-aware queries can
filter on `prompt_version` prefix when accuracy actually matters:

    Provenance(
        source_images=[...detected from result.json...],
        overlay_refs=[],
        prompt_id="unknown",
        prompt_version="legacy_pre_v1",
        provider_config={},
        input_hash="",
    )

Idempotent: re-running on already-migrated files is a no-op.

Usage:
  python scripts/migrate_anchor_provenance_legacy.py            # dry-run
  python scripts/migrate_anchor_provenance_legacy.py --write    # write changes
  python scripts/migrate_anchor_provenance_legacy.py --paths data/areas/foo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")

LEGACY_MARKER = {
    "source_images": [],
    "overlay_refs": [],
    "prompt_id": "unknown",
    "prompt_version": "legacy_pre_v1",
    "provider_config": {},
    "input_hash": "",
}


def _is_legacy_marker(prov: dict | None) -> bool:
    return isinstance(prov, dict) and prov.get("prompt_version") == "legacy_pre_v1"


def _looks_like_anchor(d: dict) -> bool:
    """An AnchorStructure dump always has these three fields."""
    return all(k in d for k in ("anchor_id", "structure_type", "anchor_center_latlon"))


def _backfill_anchor(anchor: dict) -> bool:
    """Mutate `anchor` in place. Return True if changed."""
    changed = False

    if "provenance" not in anchor or not isinstance(anchor.get("provenance"), dict):
        marker = dict(LEGACY_MARKER)
        # Lift source_images from the legacy `source_images_used` field so we
        # don't lose the file references the old schema kept.
        sources = anchor.get("source_images_used") or []
        if isinstance(sources, list):
            marker["source_images"] = list(sources)
        anchor["provenance"] = marker
        changed = True

    if "state" not in anchor:
        anchor["state"] = "draft"
        changed = True
    if "phase_history" not in anchor:
        anchor["phase_history"] = []
        changed = True
    if "findings" not in anchor:
        anchor["findings"] = []
        changed = True
    if "seed_z18_fetch_plan" not in anchor:
        anchor["seed_z18_fetch_plan"] = None
        changed = True
    if "priority_rank" not in anchor:
        anchor["priority_rank"] = None
        changed = True
    if "zone_id" not in anchor:
        anchor["zone_id"] = None
        changed = True

    return changed


def _walk_for_anchors(node):
    """Yield every dict in `node` that looks like an AnchorStructure dump."""
    if isinstance(node, dict):
        if _looks_like_anchor(node):
            yield node
        for v in node.values():
            yield from _walk_for_anchors(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_for_anchors(item)


def _migrate_file(path: Path, write: bool) -> tuple[bool, int]:
    """Return (file_changed, num_anchors_touched)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  SKIP {path}: {exc}")
        return False, 0

    touched = 0
    for anchor in _walk_for_anchors(payload):
        if _is_legacy_marker(anchor.get("provenance")):
            continue
        if _backfill_anchor(anchor):
            touched += 1

    if not touched:
        return False, 0

    if write:
        # Save .bak once, then overwrite atomically via tmp -> replace.
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    return True, touched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true",
        help="Apply changes. Without this, runs as a dry-run and prints what would change.",
    )
    parser.add_argument(
        "--paths", nargs="*",
        help="Restrict to specific dirs/files. Defaults to scanning data/areas/.",
    )
    args = parser.parse_args()

    if args.paths:
        roots = [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in args.paths]
    else:
        roots = [REPO_ROOT / "data" / "areas"]

    targets: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix == ".json":
                targets.append(root)
            continue
        if not root.exists():
            print(f"path not found: {root}", file=sys.stderr)
            continue
        # Conservative: only result.json under */structures/*/. Other JSON files
        # in data/ are different schemas (registry.json, parsed.json from raw
        # LLM output, gt_anchors.json, etc.) and don't carry AnchorStructure dumps.
        targets.extend(root.glob("**/structures/**/result.json"))

    if not targets:
        print("no candidate files found.")
        return 0

    print(f"scanning {len(targets)} candidate file(s) (write={args.write})")
    total_files = 0
    total_anchors = 0
    for path in targets:
        changed, touched = _migrate_file(path, write=args.write)
        if changed:
            total_files += 1
            total_anchors += touched
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                rel = path
            print(f"  {'WROTE' if args.write else 'WOULD-WRITE'} {rel}  "
                  f"(+{touched} anchor backfill)")

    print()
    print(f"summary: {total_files} file(s) needed migration, "
          f"{total_anchors} anchor(s) backfilled")
    if not args.write and total_files > 0:
        print("(dry-run; rerun with --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
