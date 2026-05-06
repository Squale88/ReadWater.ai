"""CLI for deterministic CV-based area discovery.

The real implementation lives in ``readwater.pipeline.cv.discovery``;
this shim wires up env loading, CLI args, and a comparison pass against
the existing manifest.

Usage:
  python scripts/cv_discover.py --area rookery_bay_v2
  python scripts/cv_discover.py --area rookery_bay_v2 --no-compare
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

from readwater import storage  # noqa: E402
from readwater.areas import Area  # noqa: E402
from readwater.pipeline.cv.discovery import (  # noqa: E402
    discover_area,
    write_result,
)


def _compare_against_manifest(area_id: str, kept: list[str]) -> None:
    """Print a side-by-side comparison of rubric-kept cells vs LLM-kept cells.

    LLM-kept cells = those currently in the area's manifest (the existing
    100 z16 cells from the prior LLM-driven discovery run).
    """
    manifest_path = storage.area_manifest_path(area_id)
    if not manifest_path.exists():
        print(f"\n(no existing manifest at {manifest_path}; skipping comparison)")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    llm = sorted(manifest.get("cells", {}).keys())
    rubric = sorted(kept)
    both = set(llm) & set(rubric)
    rubric_only = sorted(set(rubric) - set(llm))
    llm_only = sorted(set(llm) - set(rubric))

    print()
    print("=" * 70)
    print("Comparison vs existing manifest (LLM-driven discovery output)")
    print("=" * 70)
    print(f"  Rubric kept:  {len(rubric):>3}")
    print(f"  LLM kept:     {len(llm):>3}")
    print(f"  Both:         {len(both):>3}")
    print(f"  Rubric only:  {len(rubric_only):>3}  (would have been LLM-dropped)")
    print(f"  LLM only:     {len(llm_only):>3}  (LLM-kept that rubric drops)")

    # For LLM-only cells, look up CV anchor counts to assess noise
    if llm_only:
        print()
        print("LLM-only cells with CV anchor counts (low counts = LLM was wrong):")
        zero_anchor = []
        low_anchor = []
        high_anchor = []
        for cid in llm_only:
            entry = manifest["cells"].get(cid, {})
            anchors_path = entry.get("anchors")
            if not anchors_path:
                continue
            try:
                d = json.loads((storage.data_root() / anchors_path).read_text())
                n = d.get("anchor_count", 0)
                if n == 0:
                    zero_anchor.append(cid)
                elif n <= 5:
                    low_anchor.append((cid, n))
                else:
                    high_anchor.append((cid, n))
            except Exception:
                pass
        if zero_anchor:
            print(f"  zero CV anchors  ({len(zero_anchor)}): {zero_anchor}")
        if low_anchor:
            print(f"  1-5 CV anchors   ({len(low_anchor)}): "
                  + ", ".join(f"{c}({n})" for c, n in low_anchor))
        if high_anchor:
            print(f"  6+ CV anchors    ({len(high_anchor)}): "
                  + ", ".join(f"{c}({n})" for c, n in high_anchor))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default="rookery_bay_v2",
                        help="Area id (default: rookery_bay_v2).")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for fetched styled tiles (default: "
                             "data/areas/<area>/_discovery_cache).")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the comparison pass against the existing manifest.")
    args = parser.parse_args()

    cache_dir = (Path(args.cache_dir) if args.cache_dir
                 else storage.area_root(args.area) / "_discovery_cache")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = storage.area_root(args.area) / f"cv_discovery_{ts}.json"

    print(f"Area:    {args.area}")
    print(f"Cache:   {cache_dir}")
    print(f"Output:  {output_path}")
    print()

    result = discover_area(args.area, cache_dir=cache_dir, logger=print)
    write_result(result, output_path)

    print(f"\nDiscovered {len(result.kept_z16_cells)} z16 cells -> {output_path.name}")

    if not args.no_compare:
        _compare_against_manifest(args.area, result.kept_z16_cells)

    return 0


if __name__ == "__main__":
    sys.exit(main())
