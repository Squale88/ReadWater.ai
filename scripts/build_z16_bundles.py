"""Generate context_bundle.json + overlays for arbitrary z16 cells.

Generalizes `run_retained_context_test.py` to handle any (parent, child)
pair. Uses pre-fetched imagery from data/areas/rookery_bay_v2/images/
and the canonical `assemble_z16_bundle` + `persist_bundle` helpers, so
the output matches the format the v3 anchor-identification harness
already consumes.

CONFIG below lists the cells to build. Already-built cells (existing
context_bundle.json) are skipped to keep API costs down. Use --force to
rebuild.

Usage:
  python scripts/build_z16_bundles.py
  python scripts/build_z16_bundles.py --force
  python scripts/build_z16_bundles.py --only root-2-9 root-6-10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_env = REPO_ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if not os.environ.get(_k):
            os.environ[_k] = _v

sys.path.insert(0, str(REPO_ROOT / "src"))

from readwater.api.claude_vision import MODEL, _extract_json_from_response  # noqa: E402,F401
from readwater.models.cell import BoundingBox  # noqa: E402
from readwater.models.context import CellContext, LineageRef  # noqa: E402
from readwater.pipeline.cell_analyzer import (  # noqa: E402
    _sub_cell_bbox,
    _subdivide_bbox,
    ground_coverage_miles,
)
from readwater.pipeline.context_bundle import (  # noqa: E402
    assemble_z16_bundle,
    build_cell_context,
    bundle_path_for,
    persist_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_z16_bundles")

AREA_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2"
IMG_DIR = AREA_DIR / "images"
METADATA_PATH = IMG_DIR / "metadata.json"

# What to build. (parent_id, [child_cell_nums])
# These complete the GT-anchor coverage for the 6 cells in
# ground_truth/anchors/. Only the (parent, child) pairs that don't
# already have a bundle on disk get processed.
CONFIG: list[tuple[str, list[int]]] = [
    ("root-2", [9]),
    ("root-6", [10]),
    ("root-7", [14]),
    ("root-11", [1]),
]


def _cell_num_to_row_col(n: int) -> tuple[int, int]:
    return ((n - 1) // 4, (n - 1) % 4)


def _bbox_from_meta(entry: dict) -> BoundingBox:
    b = entry["bbox"]
    return BoundingBox(north=b["north"], south=b["south"], east=b["east"], west=b["west"])


def _lineage_ref_from_meta(entry: dict, position_in_parent: tuple[int, int] | None) -> LineageRef:
    image_path = entry["provider_images"]["google_static"]
    return LineageRef(
        cell_id=entry["cell_id"],
        zoom=entry["zoom"],
        depth=entry["depth"],
        center=(entry["center"][0], entry["center"][1]),
        bbox=_bbox_from_meta(entry),
        image_path=str(REPO_ROOT / image_path),
        position_in_parent=position_in_parent,
    )


def _parse_grid_markdown(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return _extract_json_from_response(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse grid markdown %s: %s", path.name, exc)
        return {}


def _merge_dual_pass_digest(yes_path: Path, no_path: Path) -> dict:
    yes = _parse_grid_markdown(yes_path)
    no = _parse_grid_markdown(no_path)
    if not yes and not no:
        return {}
    y_scores = {int(s["cell_number"]): float(s["score"]) for s in yes.get("sub_scores", [])}
    n_scores = {int(s["cell_number"]): float(s["score"]) for s in no.get("sub_scores", [])}
    merged: list[dict] = []
    for cn in range(1, 17):
        ky = y_scores.get(cn, 0.0) >= 4
        kn = n_scores.get(cn, 0.0) >= 4
        score = 5.0 if (ky and kn) else (0.0 if (not ky and not kn) else 3.0)
        merged.append({"cell_number": cn, "score": score})
    return {
        "summary": yes.get("summary", "") or no.get("summary", ""),
        "hydrology_notes": yes.get("hydrology_notes", "") or no.get("hydrology_notes", ""),
        "sub_scores": merged,
    }


def _grid_digest_for(cell_id: str) -> dict:
    if cell_id == "root":
        stem = "z0"
    else:
        nums = cell_id.removeprefix("root-").replace("-", "_")
        stem = f"z0_{nums}"
    yes = IMG_DIR / f"{stem}_grid_yes.md"
    no = IMG_DIR / f"{stem}_grid_no.md"
    return _merge_dual_pass_digest(yes, no)


async def _build_root_ctx(meta_by_id: dict, model: str):
    root_meta = meta_by_id["root"]
    root_lineage = _lineage_ref_from_meta(root_meta, position_in_parent=None)
    logger.info("[root] build_cell_context (z12)")
    root_ctx = await build_cell_context(
        cell_id="root",
        zoom=12,
        image_path=root_lineage.image_path,
        center=root_lineage.center,
        coverage_miles=root_meta["size_miles"],
        ancestor_lineage=[],
        ancestor_contexts={},
        grid_scoring_result=_grid_digest_for("root"),
        model_used=model,
    )
    return root_lineage, root_ctx


async def _build_parent_ctx(parent_id: str, meta_by_id: dict,
                             root_lineage, root_ctx, model: str):
    parent_meta = meta_by_id[parent_id]
    parent_num = int(parent_id.removeprefix("root-"))
    position = _cell_num_to_row_col(parent_num)
    parent_lineage = _lineage_ref_from_meta(parent_meta, position_in_parent=position)
    logger.info("[%s] build_cell_context (z14)", parent_id)
    parent_ctx = await build_cell_context(
        cell_id=parent_id,
        zoom=14,
        image_path=parent_lineage.image_path,
        center=parent_lineage.center,
        coverage_miles=parent_meta["size_miles"],
        ancestor_lineage=[root_lineage],
        ancestor_contexts={"root": root_ctx},
        grid_scoring_result=_grid_digest_for(parent_id),
        model_used=model,
    )
    return parent_lineage, parent_ctx


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if context_bundle.json already exists.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Limit to specific child cell_ids (e.g. root-2-9 root-6-10).")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.")
        return 2

    meta_list = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    meta_by_id = {e["cell_id"]: e for e in meta_list}

    # Determine which (parent, child) pairs actually need work.
    work: list[tuple[str, int, str]] = []  # (parent_id, child_num, child_id)
    for parent_id, child_nums in CONFIG:
        if parent_id not in meta_by_id:
            logger.warning("parent %s not in metadata; skipping", parent_id)
            continue
        for cn in child_nums:
            child_id = f"{parent_id}-{cn}"
            if args.only and child_id not in args.only:
                continue
            bundle_path = IMG_DIR / "structures" / child_id / "context_bundle.json"
            if bundle_path.exists() and not args.force:
                logger.info("skip %s (bundle exists; use --force to rebuild)", child_id)
                continue
            work.append((parent_id, cn, child_id))

    if not work:
        logger.info("nothing to do.")
        return 0

    # Build root once, then each parent once, sharing across its children.
    root_lineage, root_ctx = await _build_root_ctx(meta_by_id, MODEL)

    parents_done: dict[str, tuple[object, object]] = {}
    bundles_written: list[Path] = []

    for parent_id, cell_num, child_id in work:
        if parent_id not in parents_done:
            parents_done[parent_id] = await _build_parent_ctx(
                parent_id, meta_by_id, root_lineage, root_ctx, MODEL,
            )
        parent_lineage, parent_ctx = parents_done[parent_id]

        parent_meta = meta_by_id[parent_id]
        parent_bbox = _bbox_from_meta(parent_meta)
        row, col = _cell_num_to_row_col(cell_num)
        subs = _subdivide_bbox(parent_bbox, sections=4)
        center_lookup = {(r, c): ctr for r, c, ctr in subs}
        z16_center = center_lookup[(row, col)]
        z16_bbox = _sub_cell_bbox(parent_bbox, row, col, sections=4)
        z16_size_miles = ground_coverage_miles(16, z16_center[0])

        parent_num_str = parent_id.removeprefix("root-")
        z16_img = IMG_DIR / f"z0_{parent_num_str}_{cell_num}.png"
        z15_ctx_img = IMG_DIR / f"z0_{parent_num_str}_{cell_num}_context_z15.png"
        if not z16_img.exists():
            logger.warning("z16 image missing for %s: %s — skipping", child_id, z16_img)
            continue

        self_lineage = LineageRef(
            cell_id=child_id, zoom=16, depth=2,
            center=z16_center, bbox=z16_bbox,
            image_path=str(z16_img),
            position_in_parent=(row, col),
        )
        logger.info("[%s] build_cell_context (z16)", child_id)
        z16_ctx = await build_cell_context(
            cell_id=child_id, zoom=16,
            image_path=str(z16_img),
            center=z16_center,
            coverage_miles=z16_size_miles,
            ancestor_lineage=[root_lineage, parent_lineage],
            ancestor_contexts={"root": root_ctx, parent_id: parent_ctx},
            grid_scoring_result=_grid_digest_for(child_id),
            model_used=MODEL,
        )

        bundle = assemble_z16_bundle(
            self_lineage=self_lineage,
            self_context=z16_ctx,
            ancestor_lineage=[root_lineage, parent_lineage],
            ancestor_contexts={"root": root_ctx, parent_id: parent_ctx},
            z15_same_center_path=str(z15_ctx_img) if z15_ctx_img.exists() else None,
            base_output_dir=str(IMG_DIR),
        )
        bundle_path = persist_bundle(bundle, bundle_path_for(str(IMG_DIR), child_id))
        bundles_written.append(Path(bundle_path))
        logger.info("  bundle written: %s", bundle_path)

    print()
    print("=" * 70)
    print(f"Bundles written: {len(bundles_written)}")
    for p in bundles_written:
        print(f"  {p.relative_to(REPO_ROOT)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
