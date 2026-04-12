"""Claude API integration for satellite image analysis."""

from __future__ import annotations

import base64
import os

import anthropic

from readwater.models.cell import CellAnalysis, CellScore

MODEL = "claude-sonnet-4-20250514"


def _get_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


def _build_analysis_prompt(parent_context: str, sections: int) -> str:
    """Build the prompt that instructs Claude to analyze a satellite image."""
    grid_label = f"{sections}x{sections}"
    total = sections * sections

    context_block = ""
    if parent_context:
        context_block = (
            f"\n\nParent cell context (the broader area this cell sits within):\n"
            f"{parent_context}\n"
        )

    return f"""You are analyzing a satellite image of an inshore saltwater coastal area
for fishing potential. Your goal is to identify fishable structure and rate each
section of the image.{context_block}

Divide this image into a {grid_label} grid ({total} cells). Row 0 is the top,
column 0 is the left.

For each cell, provide:
- A score from 0 to 10 for inshore fishing potential:
  0 = open ocean, developed land, or completely inaccessible
  3 = some water but no visible structure
  5 = moderate structure (depth changes, shoreline features)
  7 = good structure (grass flats, oyster bars, channels)
  10 = prime structure (convergence of multiple features, current seams, etc.)
- A brief summary of what you observe

Also identify any fishable structure types you see across the entire image:
grass flats, oyster bars, channels, mangrove shorelines, sand holes,
depth transitions, current seams, dock/bridge structure, etc.

Respond with valid JSON matching this schema:
{{
  "overall_summary": "string — high-level description of the area",
  "sub_scores": [
    {{"row": int, "col": int, "score": float, "summary": "string"}}
  ],
  "structure_types": ["string"]
}}"""


async def analyze_image_with_claude(
    image_data: bytes,
    parent_context: str = "",
    sections: int = 3,
    sub_cell_centers: list[tuple[float, float]] | None = None,
) -> CellAnalysis:
    """Send a satellite image to Claude for fishing potential analysis.

    Args:
        image_data: Raw PNG image bytes.
        parent_context: Summary from the parent cell for geographic continuity.
        sections: Grid divisions per side (3 = 3x3 grid).
        sub_cell_centers: Pre-computed (lat, lon) centers for each sub-cell,
            ordered row-major. If None, centers won't be populated in scores.

    Returns:
        CellAnalysis with scores and summaries for each sub-cell.
    """
    client = _get_client()
    prompt = _build_analysis_prompt(parent_context, sections)
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
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
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    # TODO: Parse Claude's JSON response into CellAnalysis
    # For now, return a placeholder
    raw_text = response.content[0].text
    usage = response.usage

    # Parsing logic will go here — extract JSON from raw_text,
    # map sub_cell_centers onto each CellScore, and build the CellAnalysis

    raise NotImplementedError(
        "JSON response parsing not yet implemented. "
        f"Raw response length: {len(raw_text)} chars"
    )
