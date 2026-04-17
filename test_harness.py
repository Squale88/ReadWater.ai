"""Test harness for iterating on grid scoring prompts.

Uses existing images in data/areas/rookery_bay_test/images/ with hand-labeled
ground truth. Runs dual_pass_grid_scoring on each image and evaluates.

Ground truth rules:
- MUST_KEEP: cell must end up KEEP (score >= 4)
- MUST_PRUNE: cell must end up PRUNE (score < 3)
- FLEX: either KEEP, AMBIG, or PRUNE is acceptable
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from readwater.api.claude_vision import dual_pass_grid_scoring  # noqa

IMG_DIR = Path("data/areas/rookery_bay_test/images")


@dataclass
class TestCase:
    name: str
    image_file: str
    center: tuple[float, float]
    zoom: int
    coverage_miles: float
    must_keep: set[int]
    must_prune: set[int]
    description: str

    @property
    def flex(self) -> set[int]:
        return set(range(1, 17)) - self.must_keep - self.must_prune


# --- TEST CASES (hand-labeled from visual review) ---

TEST_CASES = [
    TestCase(
        name="root_zoom12",
        image_file="z0_grid.png",
        center=(26.029727, -81.746663),
        zoom=12,
        coverage_miles=13.7,
        must_keep={2, 6, 7, 10, 11, 12, 14, 15, 16},
        must_prune={1, 3, 5, 9, 13},
        description="Root Rookery Bay at zoom 12 — the canonical test",
    ),
    TestCase(
        name="open_gulf_zoom16",
        image_file="z0_14_11_google_static_grid.png",
        center=(25.949316557137617, -81.76731295143725),
        zoom=16,
        coverage_miles=0.85,
        must_keep=set(),
        must_prune=set(range(1, 17)),  # ALL 16 must prune
        description="Pure open Gulf at zoom 16 with stitching artifact — CRITICAL",
    ),
    TestCase(
        name="mangrove_creek_zoom16",
        image_file="z0_16_11_google_static_grid.png",
        center=(25.949316557137617, -81.65717987710528),
        zoom=16,
        coverage_miles=0.85,
        must_keep={11, 12, 13, 14, 15, 16},  # clear creek + mangrove edges
        must_prune=set(),
        description="Mangrove creek with tidal channels at zoom 16",
    ),
    TestCase(
        name="marco_residential_zoom14",
        image_file="z0_15_grid.png",
        center=(25.95550558301288, -81.719129731417),  # root-15
        zoom=14,
        coverage_miles=3.41,
        must_keep={1, 2, 3, 4, 8},  # open bay water with islands
        must_prune={13, 14},  # beach and dense residential
        description="Marco Island residential + bay at zoom 14",
    ),
    TestCase(
        name="mangrove_backcountry_zoom14",
        image_file="z0_11_grid.png",
        center=(26.004986527670958, -81.719129731417),  # root-11
        zoom=14,
        coverage_miles=3.41,
        must_keep={1, 2, 5, 6, 9, 10, 11, 13, 15},  # mangrove backcountry + shallow flats
        must_prune=set(),
        description="Rookery Bay mangrove backcountry at zoom 14",
    ),
    TestCase(
        name="barrier_island_zoom14",
        image_file="z0_10_grid.png",
        center=(26.004986527670958, -81.77419626858298),  # root-10
        zoom=14,
        coverage_miles=3.41,
        must_keep={3, 4, 7, 8, 11, 12, 16},  # barrier island + bay side (cell 15 is flex — mostly water, thin sand strip)
        must_prune={1, 2, 5, 6, 9, 10, 13},  # open Gulf
        description="Barrier island + Gulf at zoom 14 (Keewaydin)",
    ),
]


def parse_scores_from_md(md_path: Path) -> dict[int, float] | None:
    """Extract scores dict from a saved raw response .md file."""
    if not md_path.exists():
        return None
    text = md_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
    if not matches:
        return None
    try:
        data = json.loads(matches[-1].group(1).strip())
        return {sc["cell_number"]: float(sc["score"]) for sc in data.get("sub_scores", [])}
    except Exception:
        return None


async def _fetch_context_image(tc: TestCase) -> str | None:
    """Fetch a zoom-1 context image for the test case if we don't have it."""
    import os
    from readwater.api.providers.google_static import GoogleStaticProvider

    context_path = IMG_DIR / f"{Path(tc.image_file).stem}_context_z{tc.zoom-1}.png"
    if context_path.exists():
        return str(context_path)
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        return None
    provider = GoogleStaticProvider()
    try:
        await provider.fetch(tc.center, tc.zoom - 1, str(context_path))
        return str(context_path)
    except Exception as e:
        print(f"  context fetch failed: {e}")
        return None


async def run_test_case(tc: TestCase, use_context: bool = True) -> dict:
    """Run dual_pass_grid_scoring on a test image and evaluate."""
    image_path = str(IMG_DIR / tc.image_file)
    context_image_path = await _fetch_context_image(tc) if use_context else None
    result = await dual_pass_grid_scoring(
        image_path, "", tc.zoom, tc.center, tc.coverage_miles,
        context_image_path=context_image_path,
    )

    # Save raw responses with a suffix so we don't overwrite the originals
    suffix = f".harness.md"
    raw_yes = result.get("raw_response_yes", "")
    raw_no = result.get("raw_response_no", "")
    if raw_yes:
        (IMG_DIR / f"{Path(tc.image_file).stem}_yes{suffix}").write_text(raw_yes, encoding="utf-8")
    if raw_no:
        (IMG_DIR / f"{Path(tc.image_file).stem}_no{suffix}").write_text(raw_no, encoding="utf-8")

    scores = {sc["cell_number"]: float(sc["score"]) for sc in result.get("sub_scores", [])}

    # Evaluate
    verdicts = {}
    for z in range(1, 17):
        s = scores.get(z, 0)
        if s >= 4:
            verdicts[z] = "KEEP"
        elif s >= 3:
            verdicts[z] = "AMBIG"
        else:
            verdicts[z] = "PRUNE"

    # AMBIG cells trigger confirmation check in the pipeline, so they're not lost.
    # A true false negative is a must-keep cell that gets PRUNE (no confirmation).
    # A true false positive is a must-prune cell that gets KEEP (recurses without check).
    false_neg = sorted(z for z in tc.must_keep if verdicts[z] == "PRUNE")
    false_pos = sorted(z for z in tc.must_prune if verdicts[z] == "KEEP")
    borderline_misses = sorted(z for z in tc.must_prune if verdicts[z] == "AMBIG")
    borderline_lost = sorted(z for z in tc.must_keep if verdicts[z] == "AMBIG")

    return {
        "test": tc.name,
        "scores": scores,
        "verdicts": verdicts,
        "false_neg": false_neg,
        "false_pos": false_pos,
        "borderline_misses": borderline_misses,
        "borderline_lost": borderline_lost,
        "passed": len(false_neg) == 0 and len(false_pos) == 0,
    }


def print_result(tc: TestCase, result: dict):
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n=== {tc.name} [{status}] ===")
    print(f"  {tc.description}")
    print(f"  MUST-KEEP: {sorted(tc.must_keep)}")
    print(f"  MUST-PRUNE: {sorted(tc.must_prune)}")
    verdict_str = " ".join(
        f"{z}:{result['verdicts'][z][0]}" for z in range(1, 17)
    )
    print(f"  Results: {verdict_str}")
    if result["false_neg"]:
        print(f"  FALSE NEG (must-keep but not kept): {result['false_neg']}")
    if result["false_pos"]:
        print(f"  FALSE POS (must-prune but kept): {result['false_pos']}")
    if result["borderline_lost"]:
        print(f"  Must-keep gone ambiguous: {result['borderline_lost']}")


async def main():
    results = []
    for tc in TEST_CASES:
        print(f"Running {tc.name}...", end="", flush=True)
        result = await run_test_case(tc)
        results.append((tc, result))
        print(" done")

    for tc, result in results:
        print_result(tc, result)

    passed = sum(1 for _, r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
