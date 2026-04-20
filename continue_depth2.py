"""Continue a Rookery Bay v2 test from depth 1 to depth 2.

Reads existing depth-1 grid scoring results from disk and drives the
depth-2 work directly: for each depth-1 cell's sub-cells that scored
KEEP or AMBIG, fetch the depth-2 image and run dual-pass scoring.
AMBIG cells go through confirmation first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

from readwater.api.claude_vision import (  # noqa
    confirm_fishing_water,
    dual_pass_grid_scoring,
)
from readwater.api.providers.google_static import GoogleStaticProvider  # noqa
from readwater.pipeline.cell_analyzer import (  # noqa
    EARTH_CIRCUMFERENCE_MILES,
    MILES_PER_DEG_LAT,
    _role_for_zoom,
    ground_coverage_miles,
)
from readwater.pipeline.image_processing import draw_grid_overlay  # noqa


IMG_DIR = Path("data/areas/rookery_bay_v2/images")
ROOT_CENTER = (26.029727, -81.746663)
ROOT_ZOOM = 12


def _cell_center(parent_center: tuple[float, float], parent_coverage: float, cell_num: int):
    """Compute the geographic center of cell N (1-16) within a parent 4x4 grid."""
    row = (cell_num - 1) // 4
    col = (cell_num - 1) % 4
    cell_miles = parent_coverage / 4
    half_lat = (parent_coverage / 2) / MILES_PER_DEG_LAT
    parent_north = parent_center[0] + half_lat
    cos_lat = math.cos(math.radians(parent_center[0]))
    half_lon = (parent_coverage / 2) / (MILES_PER_DEG_LAT * cos_lat)
    parent_west = parent_center[1] - half_lon
    cell_lat_deg = cell_miles / MILES_PER_DEG_LAT
    cell_lon_deg = cell_miles / (MILES_PER_DEG_LAT * cos_lat)
    center_lat = parent_north - (row + 0.5) * cell_lat_deg
    center_lon = parent_west + (col + 0.5) * cell_lon_deg
    return (center_lat, center_lon)


def _parse_scores(md_path: Path) -> dict[int, float]:
    """Extract {cell_number: score} dict from a saved raw response .md file."""
    if not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
    if not matches:
        return {}
    try:
        data = json.loads(matches[-1].group(1).strip())
        return {sc["cell_number"]: float(sc["score"]) for sc in data.get("sub_scores", [])}
    except Exception:
        return {}


def _merge_dual(yes: dict[int, float], no: dict[int, float]) -> dict[int, float]:
    """Merge YES/NO pass scores into final 5/3/0 scale."""
    merged = {}
    for cn in range(1, 17):
        ky = yes.get(cn, 0) >= 4
        kn = no.get(cn, 0) >= 4
        if ky and kn:
            merged[cn] = 5.0
        elif not ky and not kn:
            merged[cn] = 0.0
        else:
            merged[cn] = 3.0
    return merged


async def _retry(coro_fn, label: str, retries: int = 5, base_delay: float = 2.0):
    """Retry an async call with exponential backoff on network errors."""
    for i in range(retries):
        try:
            return await coro_fn()
        except Exception as e:
            if i == retries - 1:
                raise
            delay = base_delay * (2 ** i)
            print(f"  {label} failed ({type(e).__name__}), retry {i+1}/{retries} in {delay:.0f}s...")
            await asyncio.sleep(delay)
    return None


async def process_depth2_for_parent(parent_id: str, parent_center: tuple, parent_zoom: int,
                                    parent_summary: str, provider: GoogleStaticProvider,
                                    api_budget: list[int], max_calls: int) -> dict:
    """Process depth-2 work for a single depth-1 parent cell.

    Reads the parent's grid scoring, then for each qualifying sub-cell (score 5 or 3):
      - Fetch the child image at zoom 16
      - If ambiguous: run confirmation
      - If kept or confirmed: fetch context image, draw grid, run dual-pass
    Returns a summary dict.
    """
    parent_stem = f"z0_{parent_id.removeprefix('root-')}" if parent_id != "root" else "z0"
    parent_yes = _parse_scores(IMG_DIR / f"{parent_stem}_grid_yes.md")
    parent_no = _parse_scores(IMG_DIR / f"{parent_stem}_grid_no.md")
    merged = _merge_dual(parent_yes, parent_no)

    parent_coverage = ground_coverage_miles(parent_zoom, parent_center[0])
    child_zoom = parent_zoom + 2
    context_zoom = child_zoom - 1

    summary = {"parent": parent_id, "kept": [], "confirmed_ambig": [], "rejected_ambig": [],
               "pruned": [], "grid_scored": []}

    for cell_num, score in merged.items():
        if api_budget[0] >= max_calls:
            break
        if score < 3:
            summary["pruned"].append(cell_num)
            continue

        child_center = _cell_center(parent_center, parent_coverage, cell_num)
        child_size = ground_coverage_miles(child_zoom, child_center[0])
        child_id = f"{parent_id}-{cell_num}"
        child_stem = f"{parent_stem}_{cell_num}"

        # Fetch primary child image (with retry)
        primary_path = IMG_DIR / f"{child_stem}.png"
        if not primary_path.exists():
            try:
                await _retry(
                    lambda: provider.fetch(child_center, child_zoom, str(primary_path)),
                    f"fetch {child_id}",
                )
                api_budget[0] += 1
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"  SKIP {child_id}: fetch failed after retries: {e}")
                continue

        # Ambiguous → confirmation first
        if score == 3:
            if api_budget[0] >= max_calls:
                break
            try:
                confirm = await _retry(
                    lambda: confirm_fishing_water(
                        str(primary_path), parent_summary, child_center, child_size,
                    ),
                    f"confirm {child_id}",
                )
            except Exception as e:
                print(f"  SKIP {child_id}: confirmation failed: {e}")
                continue
            api_budget[0] += 1
            raw = confirm.pop("raw_response", "")
            if raw:
                (IMG_DIR / f"{child_stem}_confirm.md").write_text(raw, encoding="utf-8")
            if not confirm.get("has_fishing_water", False):
                summary["rejected_ambig"].append(cell_num)
                continue
            summary["confirmed_ambig"].append(cell_num)
        else:
            summary["kept"].append(cell_num)

        # Fetch context image (zoom-1) with retry
        context_path = IMG_DIR / f"{child_stem}_context_z{context_zoom}.png"
        if not context_path.exists():
            if api_budget[0] >= max_calls:
                break
            try:
                await _retry(
                    lambda: provider.fetch(child_center, context_zoom, str(context_path)),
                    f"context {child_id}",
                )
                api_budget[0] += 1
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"  SKIP context for {child_id}: {e}")
                # Continue without context image

        # Grid overlay
        grid_path = IMG_DIR / f"{child_stem}_grid.png"
        if not grid_path.exists():
            draw_grid_overlay(str(primary_path), output_path=str(grid_path))

        # Dual-pass grid scoring with retry
        if api_budget[0] + 2 > max_calls:
            break
        try:
            result = await _retry(
                lambda: dual_pass_grid_scoring(
                    str(grid_path), parent_summary, child_zoom, child_center, child_size,
                    context_image_path=str(context_path) if context_path.exists() else None,
                ),
                f"grid-score {child_id}",
            )
        except Exception as e:
            print(f"  SKIP grid scoring for {child_id}: {e}")
            continue
        api_budget[0] += 2  # two Claude calls

        raw_yes = result.pop("raw_response_yes", "")
        raw_no = result.pop("raw_response_no", "")
        if raw_yes:
            (IMG_DIR / f"{child_stem}_grid_yes.md").write_text(raw_yes, encoding="utf-8")
        if raw_no:
            (IMG_DIR / f"{child_stem}_grid_no.md").write_text(raw_no, encoding="utf-8")

        # Count sub-scoring breakdown
        k = sum(1 for sc in result.get("sub_scores", []) if sc["score"] >= 4)
        a = sum(1 for sc in result.get("sub_scores", []) if 3 <= sc["score"] < 4)
        p = sum(1 for sc in result.get("sub_scores", []) if sc["score"] < 3)
        summary["grid_scored"].append({
            "cell_num": cell_num, "child_id": child_id, "kept": k, "ambig": a, "pruned": p,
        })

    return summary


async def main():
    provider = GoogleStaticProvider()
    api_budget = [0]
    max_calls = 300  # generous — depth-1 already used ~30

    # Derive depth-1 parents with their centers
    root_coverage = ground_coverage_miles(ROOT_ZOOM, ROOT_CENTER[0])
    root_yes = _parse_scores(IMG_DIR / "z0_grid_yes.md")
    root_no = _parse_scores(IMG_DIR / "z0_grid_no.md")
    root_merged = _merge_dual(root_yes, root_no)

    # Root summary for parent_context
    root_summary_match = re.search(
        r'"summary":\s*"([^"]+)"',
        (IMG_DIR / "z0_grid_yes.md").read_text(encoding="utf-8"),
    )
    root_summary = root_summary_match.group(1) if root_summary_match else ""

    # Identify which depth-1 parents were processed (have their own grid scoring)
    parents_to_process = []
    for cell_num in range(1, 17):
        if root_merged.get(cell_num, 0) >= 4:  # root cells that passed straight to grid
            center = _cell_center(ROOT_CENTER, root_coverage, cell_num)
            parent_id = f"root-{cell_num}"
            parents_to_process.append((parent_id, center, cell_num))
        elif root_merged.get(cell_num, 0) == 3:
            # Ambiguous — check if confirmation rejected it
            confirm_md = IMG_DIR / f"z0_{cell_num}_confirm.md"
            yes_md = IMG_DIR / f"z0_{cell_num}_grid_yes.md"
            if yes_md.exists():  # was processed (confirmed + grid-scored)
                center = _cell_center(ROOT_CENTER, root_coverage, cell_num)
                parent_id = f"root-{cell_num}"
                parents_to_process.append((parent_id, center, cell_num))

    print(f"Processing depth-2 for {len(parents_to_process)} depth-1 parents...")
    print(f"API budget: {max_calls}")

    results = []
    for parent_id, center, _cn in parents_to_process:
        print(f"\n--- {parent_id} (API used: {api_budget[0]}) ---")
        if api_budget[0] >= max_calls:
            print("  API budget exhausted")
            break
        try:
            result = await process_depth2_for_parent(
                parent_id, center, ROOT_ZOOM + 2, root_summary,
                provider, api_budget, max_calls,
            )
        except Exception as e:
            print(f"  PARENT {parent_id} failed: {type(e).__name__}: {e}")
            continue
        results.append(result)
        # Checkpoint after each parent
        (IMG_DIR / "depth2_checkpoint.json").write_text(
            json.dumps({"api_calls": api_budget[0], "results": results}, indent=2),
            encoding="utf-8",
        )
        n_keep = len(result["kept"])
        n_conf = len(result["confirmed_ambig"])
        n_rej = len(result["rejected_ambig"])
        n_prune = len(result["pruned"])
        n_scored = len(result["grid_scored"])
        print(f"  Kept: {n_keep}, Confirmed ambig: {n_conf}, Rejected ambig: {n_rej}, Pruned: {n_prune}")
        print(f"  Grid scored: {n_scored} children at depth 2")

    # Totals
    print(f"\n{'='*60}")
    print(f"API calls used: {api_budget[0]} / {max_calls}")
    total_scored = sum(len(r["grid_scored"]) for r in results)
    print(f"Total depth-2 cells grid-scored: {total_scored}")

    # Save summary
    (IMG_DIR / "depth2_summary.json").write_text(
        json.dumps({"api_calls": api_budget[0], "results": results}, indent=2),
        encoding="utf-8",
    )

    # File stats
    pngs = len(list(IMG_DIR.glob("*.png")))
    mds = len(list(IMG_DIR.glob("*.md")))
    total = sum(f.stat().st_size for f in IMG_DIR.iterdir())
    print(f"\nFiles now: {pngs} images, {mds} docs, {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
