"""Claude vision API integration for satellite image analysis.

Prompts are loaded from the prompts/ directory at the project root.
Responses use chain-of-thought: Claude reasons freely, then provides
a JSON block that we extract.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

# Locate prompts directory relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROMPTS_DIR = _PROJECT_ROOT / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt text file from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _get_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


def _extract_json_from_response(text: str) -> dict:
    """Extract the last JSON block from a chain-of-thought response.

    Claude reasons freely, then includes a ```json ... ``` block.
    We find and parse that block.
    """
    # Find the last ```json ... ``` block
    matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())

    # Fallback: try the last ``` ... ``` block
    matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())

    # Last resort: try parsing the whole thing as JSON
    return json.loads(text.strip())


def _cell_number_to_row_col(cell_number: int, sections: int = 4) -> tuple[int, int]:
    """Convert 1-based cell number to (row, col). Cell 1=(0,0), Cell 16=(3,3)."""
    return ((cell_number - 1) // sections, (cell_number - 1) % sections)


async def analyze_grid_image(
    image_path: str,
    parent_context: str,
    current_zoom: int,
    center: tuple[float, float],
    coverage_miles: float,
    user_prompt_file: str = "grid_scoring_user.txt",
    context_image_path: str | None = None,
) -> dict:
    """Send a grid-overlay satellite image to Claude for cell-by-cell scoring.

    Uses chain-of-thought: Claude reasons through each cell, then provides
    a JSON block with scores. Returns a dict with:
    - raw_response: str — Claude's full reasoning text
    - summary, sub_scores, hydrology_notes — parsed from the JSON block

    If context_image_path is provided, a second (wider-area) image is sent
    alongside the gridded image. The model is told to use it for context —
    particularly helpful for distinguishing interior bay water (surrounded
    by land in the wider view) from open ocean (nothing but water).
    """
    client = _get_client()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = _load_prompt("grid_scoring_system.txt")

    context_line = ""
    if parent_context:
        context_line = f"\nParent cell context:\n{parent_context}\n"

    user_template = _load_prompt(user_prompt_file)
    user_prompt = user_template.format(
        parent_context=context_line,
        current_zoom=current_zoom,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.1f}",
    )

    # Build content with optional context image first, then the gridded image.
    content = []
    if context_image_path:
        with open(context_image_path, "rb") as f:
            ctx_b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "text",
            "text": "CONTEXT IMAGE (wider area, no grid) — use ONLY to understand the surroundings of the gridded cell:",
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": ctx_b64},
        })
        content.append({
            "type": "text",
            "text": "GRIDDED IMAGE (your scoring target, 4x4 numbered grid) — score the cells in this image:",
        })
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
    })
    content.append({"type": "text", "text": user_prompt})

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


async def dual_pass_grid_scoring(
    image_path: str,
    parent_context: str,
    current_zoom: int,
    center: tuple[float, float],
    coverage_miles: float,
    context_image_path: str | None = None,
) -> dict:
    """Run YES-lean and NO-lean grid scoring passes and merge results.

    Pass 1 uses grid_scoring_user.txt (lean YES).
    Pass 2 uses grid_scoring_user2.txt (lean NO).

    Merged scores per cell:
      5 — both passes said YES (confident keep)
      3 — passes disagreed (ambiguous, needs confirmation)
      0 — both passes said NO (confident prune)

    If context_image_path is provided, a wider-area image (2x coverage) is
    sent alongside the gridded image. Helps distinguish interior bay water
    from open ocean.

    Returns a merged result dict with summary, sub_scores, hydrology_notes,
    raw_response_yes, and raw_response_no.
    """
    import json as _json

    async def _one_pass(prompt_file, label):
        for retry in range(3):
            try:
                result = await analyze_grid_image(
                    image_path, parent_context, current_zoom, center,
                    coverage_miles, user_prompt_file=prompt_file,
                    context_image_path=context_image_path,
                )
                scores = {sc["cell_number"]: float(sc["score"]) for sc in result.get("sub_scores", [])}
                if len(scores) == 16:
                    return result, scores
            except (_json.JSONDecodeError, KeyError):
                pass
        # All retries failed — return empty scores
        return {"sub_scores": [], "summary": "", "hydrology_notes": "", "raw_response": ""}, {}

    result_yes, scores_yes = await _one_pass("grid_scoring_user.txt", "YES")
    result_no, scores_no = await _one_pass("grid_scoring_user2.txt", "NO")

    # Merge: both YES → 5, both NO → 0, disagree → 3
    merged_sub_scores = []
    for cell_num in range(1, 17):
        ky = scores_yes.get(cell_num, 0) >= 4
        kn = scores_no.get(cell_num, 0) >= 4
        if ky and kn:
            score = 5.0
        elif not ky and not kn:
            score = 0.0
        else:
            score = 3.0

        # Use reasoning from whichever pass scored higher
        reasoning = ""
        for sc in result_yes.get("sub_scores", []):
            if sc.get("cell_number") == cell_num:
                reasoning = sc.get("reasoning", "")
                break

        merged_sub_scores.append({
            "cell_number": cell_num,
            "score": score,
            "reasoning": reasoning,
        })

    return {
        "summary": result_yes.get("summary", ""),
        "sub_scores": merged_sub_scores,
        "hydrology_notes": result_yes.get("hydrology_notes", ""),
        "raw_response": result_yes.get("raw_response", ""),
        "raw_response_yes": result_yes.get("raw_response", ""),
        "raw_response_no": result_no.get("raw_response", ""),
    }


async def confirm_fishing_water(
    image_path: str,
    parent_context: str,
    center: tuple[float, float],
    coverage_miles: float,
) -> dict:
    """Confirm whether a raw satellite image contains fishable inshore water.

    Used as a second-pass filter: after grid scoring flags a cell as potentially
    interesting, this function checks the zoomed-in raw image to confirm there
    is actually fishable water before committing to grid analysis.

    Returns dict with has_fishing_water (bool), reasoning (str), raw_response (str).
    """
    client = _get_client()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = _load_prompt("confirmation_system.txt")

    context_line = ""
    if parent_context:
        context_line = f"\nParent area context:\n{parent_context}\n"

    user_template = _load_prompt("confirmation_user.txt")
    user_prompt = user_template.format(
        parent_context=context_line,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.1f}",
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )

    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


# `analyze_structure_image` was deleted by Phase C TASK-7 cleanup. It used
# the legacy `structure_analysis_{system,user}.txt` prompts (also deleted)
# and had no callers anywhere in src/. Anchor discovery now goes through
# `readwater.pipeline.structure.anchor_discovery.run_anchor_discovery`.


async def generate_cell_context(
    image_path: str,
    cell_id: str,
    zoom: int,
    center: tuple[float, float],
    coverage_miles: float,
    ancestor_chain_block: str = "(none)",
    grid_scoring_digest: str = "(none)",
    open_thread_block: str = "(none)",
) -> dict:
    """Generate a structured CellContext payload for a retained cell.

    Sends the cell's image plus compact digests of its ancestor chain, its
    own grid-scoring result, and any open ancestor threads to Claude and
    returns the parsed JSON response.

    The returned dict has:
      - observations[], morphology[], feature_threads[], open_questions[]
        (raw LLM output — the caller is expected to assign deterministic
        IDs and resolve local-idx references in build_cell_context).
      - raw_response: str — Claude's full reasoning text.

    No disk I/O here; persisting the raw markdown is the caller's job.
    """
    client = _get_client()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = _load_prompt("cell_context_system.txt")
    user_template = _load_prompt("cell_context_user.txt")
    user_prompt = user_template.format(
        cell_id=cell_id,
        zoom=zoom,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.2f}",
        ancestor_chain_block=ancestor_chain_block or "(none)",
        grid_scoring_digest=grid_scoring_digest or "(none)",
        open_thread_block=open_thread_block or "(none)",
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )

    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed
