"""Live smoke test of the structure phase against real fetched imagery.

Reuses pre-fetched images from `data/areas/rookery_bay_v2/images/`:
  z0_10_8.png + z0_10_8_context_z15.png   (root-10-8)
  z0_11_5.png + z0_11_5_context_z15.png   (root-11-5)

Requires:
  - ANTHROPIC_API_KEY (loaded from project-root .env)
  - GOOGLE_MAPS_API_KEY (for zoom-18 tile fetches during the structure phase)

Outputs go to `data/areas/rookery_bay_v2_structure_test/images/cells/<cell_id>/structures/`.
Usage:
  python scripts/smoke_structure_phase.py [--cell root-10-8|root-11-5|both]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os  # noqa: E402
import sys
from pathlib import Path

# Load .env from the non-worktree project root where the keys live.
REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

# Ensure the worktree src/ is on sys.path before the non-worktree install.
WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "scripts"))

from _cells import CELLS  # noqa: E402
from readwater.api.providers.google_static import GoogleStaticProvider  # noqa: E402
from readwater.pipeline.structure.agent import StructureBudget, run_structure_phase  # noqa: E402


OUT_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai/data/areas/rookery_bay_v2_structure_test/images")


async def run_one(cell_id: str, budget: StructureBudget) -> None:
    spec = CELLS[cell_id]
    for key in ("z16", "z15"):
        if not spec[key].exists():
            raise FileNotFoundError(spec[key])

    provider = GoogleStaticProvider()
    print(f"\n=== {cell_id} ===")
    print(f"  center: {spec['center']}")
    print(f"  z15: {spec['z15'].name}")
    print(f"  z16: {spec['z16'].name}")
    print(f"  output: {OUT_ROOT / 'cells' / cell_id / 'structures'}")

    result = await run_structure_phase(
        cell_id=cell_id,
        cell_center=spec["center"],
        z15_image_path=str(spec["z15"]),
        z16_image_path=str(spec["z16"]),
        provider=provider,
        base_output_dir=OUT_ROOT,
        parent_context=spec["parent_context"],
        coverage_miles=0.37,
        budget=budget,
    )

    print(f"\n  anchors discovered: {len(result.anchors)}")
    for a in result.anchors:
        print(f"    {a.anchor_id}  {a.structure_type:15s}  scale={a.scale:5s}  "
              f"conf={a.confidence:.2f}  poly={len(a.geometry.pixel_polygon)}v")
    print(f"  deferred:   {len(result.deferred)}")
    for d in result.deferred:
        print(f"    {d.anchor_id}  {d.structure_type:15s}  rank={d.rank:.3f}")
    print(f"  complexes:  {len(result.complexes)}")
    print(f"  influences: {len(result.influences)}")
    print(f"  subzones:   {len(result.subzones)}")
    for s in result.subzones:
        print(f"    {s.subzone_id}  {s.subzone_type:25s}  prio={s.relative_priority:.2f}")
    if result.overlap_report:
        print(f"  overlap entries: {len(result.overlap_report)}")
    if result.truncated_ids:
        print(f"  truncated: {result.truncated_ids}")
    if result.failed_identifications:
        print(f"  failed_identifications: {len(result.failed_identifications)}")
        for f in result.failed_identifications:
            print(f"    {f.feature_id} ({f.feature_level}): {f.reason}")
    if result.segmentation_issues:
        print(f"  segmentation_issues: {len(result.segmentation_issues)}")
        for s in result.segmentation_issues:
            print(f"    {s.feature_id} ({s.feature_level}): {s.reason}")
    print(f"  api_calls_used: {result.api_calls_used}")
    print(f"  tiles_fetched:  {result.tiles_fetched}")
    print(f"  registry: {result.registry_path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=list(CELLS.keys()) + ["all", "both"],
        default="root-10-8",
        help="Single cell id, 'all' for every cell in _cells.CELLS, or "
             "'both' (legacy alias for the two original cells).",
    )
    parser.add_argument("--calls-per-anchor", type=int, default=10)
    parser.add_argument("--tiles-per-anchor", type=int, default=16)
    parser.add_argument("--max-anchors", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in .env)")
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        raise SystemExit("GOOGLE_MAPS_API_KEY not set (expected in .env)")

    budget = StructureBudget(
        calls_per_anchor=args.calls_per_anchor,
        tiles_per_anchor=args.tiles_per_anchor,
        max_anchors_per_cell=args.max_anchors,
    )

    if args.cell == "all":
        cells = list(CELLS.keys())
    elif args.cell == "both":
        cells = ["root-10-8", "root-11-5"]
    else:
        cells = [args.cell]
    for cid in cells:
        await run_one(cid, budget)


if __name__ == "__main__":
    asyncio.run(main())
