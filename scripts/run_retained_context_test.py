"""Populate retained-cell CellContexts + z16 handoff bundles for root-10.

Targeted Phase-1 exercise: takes the pre-fetched imagery under
`data/areas/rookery_bay_v2/images/`, reconstructs the lineage (root
at z12 -> root-10 at z14 -> retained z16 cells), and drives real
`build_cell_context` calls for each, then `assemble_z16_bundle` +
`persist_bundle` for each retained z16 cell.

Requires ANTHROPIC_API_KEY. Does NOT re-fetch any imagery.

Outputs:
  data/areas/rookery_bay_v2/images/structures/<z16_cell_id>/
      context_bundle.json
      overlay_z15_same_center.png
      overlay_z14_parent.png
      overlay_z12_grandparent.png
  data/areas/rookery_bay_v2/images/<cell>_context.md    (raw LLM responses)

Retained z16 cells under root-10 (from depth2_summary.json):
  kept:           3, 4, 7, 8, 12, 16
  confirmed ambig: 11, 15
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Bootstrap: .env + sys.path (uses Path(__file__) so it works wherever
# the repo is checked out, not a hardcoded Dropbox path).
# ------------------------------------------------------------------

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
        # Override empty/unset env vars; leave real pre-set values alone.
        if not os.environ.get(_k):
            os.environ[_k] = _v

sys.path.insert(0, str(REPO_ROOT / "src"))

# ------------------------------------------------------------------
# Imports (after path fixup).
# ------------------------------------------------------------------

from readwater.api.claude_vision import MODEL, _extract_json_from_response  # noqa: E402
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
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data locations.
# ------------------------------------------------------------------

AREA_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2"
IMG_DIR = AREA_DIR / "images"
METADATA_PATH = IMG_DIR / "metadata.json"
DEPTH2_SUMMARY_PATH = IMG_DIR / "depth2_summary.json"

# Retained z16 cell numbers under root-10 (from depth2_summary.json).
ROOT10_RETAINED_Z16_CELLS = [3, 4, 7, 8, 11, 12, 15, 16]


# ------------------------------------------------------------------
# Helpers.
# ------------------------------------------------------------------


def _cell_num_to_row_col(n: int) -> tuple[int, int]:
    """1-16 -> (row, col) in a 4x4 grid. Matches _cell_number_to_row_col."""
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
    """Extract the JSON block from a chain-of-thought grid-scoring .md file."""
    if not path.exists():
        return {}
    try:
        return _extract_json_from_response(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse grid markdown %s: %s", path.name, exc)
        return {}


def _merge_dual_pass_digest(yes_path: Path, no_path: Path) -> dict:
    """Merge the yes-lean and no-lean grid-scoring markdowns into a single
    payload shaped like dual_pass_grid_scoring's output.

    Each sub_score gets the merged score under the same rule the live
    pipeline uses: both YES -> 5, both NO -> 0, disagree -> 3.
    """
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
    """Find and merge the dual-pass grid-scoring markdowns for this cell."""
    if cell_id == "root":
        stem = "z0"
    else:
        nums = cell_id.removeprefix("root-").replace("-", "_")
        stem = f"z0_{nums}"
    yes = IMG_DIR / f"{stem}_grid_yes.md"
    no = IMG_DIR / f"{stem}_grid_no.md"
    return _merge_dual_pass_digest(yes, no)


# ------------------------------------------------------------------
# Main flow.
# ------------------------------------------------------------------


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.")
        sys.exit(2)

    # Load the metadata for root and root-10 (the z14 parent we care about).
    meta_list = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    meta_by_id = {e["cell_id"]: e for e in meta_list}
    root_meta = meta_by_id["root"]
    root10_meta = meta_by_id["root-10"]
    logger.info("root center=%s zoom=%s", root_meta["center"], root_meta["zoom"])
    logger.info("root-10 center=%s zoom=%s", root10_meta["center"], root10_meta["zoom"])

    # --- Build LineageRefs for root and root-10 ---

    root_lineage = _lineage_ref_from_meta(root_meta, position_in_parent=None)
    # root-10 is cell #10 of the root grid -> row 2, col 1.
    root10_position = _cell_num_to_row_col(10)
    root10_lineage = _lineage_ref_from_meta(root10_meta, position_in_parent=root10_position)

    # --- CellContext for root ---

    logger.info("[1/10] build_cell_context for root (z12)")
    root_ctx = await build_cell_context(
        cell_id="root",
        zoom=12,
        image_path=root_lineage.image_path,
        center=root_lineage.center,
        coverage_miles=root_meta["size_miles"],
        ancestor_lineage=[],
        ancestor_contexts={},
        grid_scoring_result=_grid_digest_for("root"),
        model_used=MODEL,
    )
    _log_context_shape("root", root_ctx)

    # --- CellContext for root-10 ---

    logger.info("[2/10] build_cell_context for root-10 (z14)")
    root10_ctx = await build_cell_context(
        cell_id="root-10",
        zoom=14,
        image_path=root10_lineage.image_path,
        center=root10_lineage.center,
        coverage_miles=root10_meta["size_miles"],
        ancestor_lineage=[root_lineage],
        ancestor_contexts={"root": root_ctx},
        grid_scoring_result=_grid_digest_for("root-10"),
        model_used=MODEL,
    )
    _log_context_shape("root-10", root10_ctx)

    # --- Process each retained z16 cell under root-10 ---

    bundles_written: list[Path] = []
    root10_bbox = _bbox_from_meta(root10_meta)

    for i, cell_num in enumerate(ROOT10_RETAINED_Z16_CELLS, start=3):
        child_id = f"root-10-{cell_num}"
        row, col = _cell_num_to_row_col(cell_num)

        # Derive z16 center + bbox from root-10 using the same math the
        # recursive pipeline uses, so the lineage geometry is consistent.
        subs = _subdivide_bbox(root10_bbox, sections=4)
        center_lookup = {(r, c): ctr for r, c, ctr in subs}
        z16_center = center_lookup[(row, col)]
        z16_bbox = _sub_cell_bbox(root10_bbox, row, col, sections=4)
        z16_size_miles = ground_coverage_miles(16, z16_center[0])

        # Filenames on disk follow the existing z0_<parentnum>_<cellnum> pattern.
        z16_img = IMG_DIR / f"z0_10_{cell_num}.png"
        z15_ctx_img = IMG_DIR / f"z0_10_{cell_num}_context_z15.png"

        if not z16_img.exists():
            logger.warning("z16 image missing for %s: %s — skipping", child_id, z16_img)
            continue

        self_lineage = LineageRef(
            cell_id=child_id,
            zoom=16,
            depth=2,
            center=z16_center,
            bbox=z16_bbox,
            image_path=str(z16_img),
            position_in_parent=(row, col),
        )

        logger.info(
            "[%d/10] build_cell_context for %s (z16, cell_num=%d, pos=%s)",
            i, child_id, cell_num, (row, col),
        )
        z16_ctx = await build_cell_context(
            cell_id=child_id,
            zoom=16,
            image_path=str(z16_img),
            center=z16_center,
            coverage_miles=z16_size_miles,
            ancestor_lineage=[root_lineage, root10_lineage],
            ancestor_contexts={"root": root_ctx, "root-10": root10_ctx},
            grid_scoring_result=_grid_digest_for(child_id),
            model_used=MODEL,
        )
        _log_context_shape(child_id, z16_ctx)

        # Assemble + persist bundle.
        bundle = assemble_z16_bundle(
            self_lineage=self_lineage,
            self_context=z16_ctx,
            ancestor_lineage=[root_lineage, root10_lineage],
            ancestor_contexts={"root": root_ctx, "root-10": root10_ctx},
            z15_same_center_path=str(z15_ctx_img) if z15_ctx_img.exists() else None,
            base_output_dir=str(IMG_DIR),
        )
        bundle_path = persist_bundle(bundle, bundle_path_for(str(IMG_DIR), child_id))
        bundles_written.append(Path(bundle_path))
        logger.info("  bundle written: %s", bundle_path)

    # --- Summary ---

    print()
    print("=" * 70)
    print(f"CellContext produced for: root, root-10, + {len(ROOT10_RETAINED_Z16_CELLS)} z16 cells")
    print(f"Z16 bundles written: {len(bundles_written)}")
    print()
    for p in bundles_written:
        print(f"  {p.relative_to(REPO_ROOT)}")
    print("=" * 70)


def _log_context_shape(cell_id: str, ctx: CellContext) -> None:
    logger.info(
        "  %s: obs=%d morph=%d threads=%d questions=%d",
        cell_id,
        len(ctx.observations),
        len(ctx.morphology),
        len(ctx.feature_threads),
        len(ctx.open_questions),
    )


if __name__ == "__main__":
    asyncio.run(main())
