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

MODEL = "claude-opus-4-20250514"
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
) -> dict:
    """Send a grid-overlay satellite image to Claude for cell-by-cell scoring.

    Uses chain-of-thought: Claude reasons through each cell, then provides
    a JSON block with scores. Returns a dict with:
    - raw_response: str — Claude's full reasoning text
    - summary, sub_scores, hydrology_notes — parsed from the JSON block
    """
    client = _get_client()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = _load_prompt("grid_scoring_system.txt")

    context_line = ""
    if parent_context:
        context_line = f"\nParent cell context:\n{parent_context}\n"

    user_template = _load_prompt("grid_scoring_user.txt")
    user_prompt = user_template.format(
        parent_context=context_line,
        current_zoom=current_zoom,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.1f}",
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


async def analyze_structure_image(
    image_path: str,
    parent_context: str,
    center: tuple[float, float],
    coverage_miles: float,
) -> dict:
    """Send a raw satellite image to Claude for detailed structure analysis.

    Uses chain-of-thought. Returns the parsed JSON plus raw_response.
    """
    client = _get_client()

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    system_prompt = _load_prompt("structure_analysis_system.txt")

    context_line = ""
    if parent_context:
        context_line = f"\nContext from parent analysis:\n{parent_context}\n"

    user_template = _load_prompt("structure_analysis_user.txt")
    user_prompt = user_template.format(
        parent_context=context_line,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.2f}",
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
