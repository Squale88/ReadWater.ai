"""Thin Claude-vision wrappers for the structure-phase prompts.

Phase 1 prompts:
  discover_anchors      — at cell level, enumerate candidate anchors with
                          rough normalized bboxes and continuation edges
  resolve_continuation  — on a mosaic, decide whether to expand tiles
  identify_anchor       — click points + influence polygon for one anchor
  identify_subzones     — click points for up to 4 subzones

Every wrapper returns the parsed JSON plus raw_response text. Coordinates
come back as normalized fractions [0, 1] of the image Claude saw, except
resolve_continuation which returns only edge booleans.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from readwater.api.claude_vision import (
    MAX_TOKENS,
    MODEL,
    _extract_json_from_response,
    _get_client,
    _load_prompt,
)

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 - 64 * 1024
_UPLOAD_JPEG_QUALITY = 85
_UPLOAD_MAX_DIM = 1800


def _image_block(path: str | Path) -> dict:
    """Re-encode an image as JPEG (with optional downscale) for Claude upload.

    Stays under Anthropic's 5 MB base64 cap by resizing or dropping quality
    as needed.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    longest = max(img.size)
    if longest > _UPLOAD_MAX_DIM:
        ratio = _UPLOAD_MAX_DIM / longest
        new_size = (int(round(img.size[0] * ratio)), int(round(img.size[1] * ratio)))
        img = img.resize(new_size, Image.LANCZOS)

    quality = _UPLOAD_JPEG_QUALITY
    data = b""
    for _ in range(4):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= _MAX_UPLOAD_BYTES:
            break
        if quality > 60:
            quality -= 10
        else:
            img = img.resize(
                (int(img.size[0] * 0.85), int(img.size[1] * 0.85)), Image.LANCZOS,
            )

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(data).decode("utf-8"),
        },
    }


# --- DISCOVER ---


async def discover_anchors(
    z15_image_path: str,
    z16_image_path: str,
    parent_context: str,
    center: tuple[float, float],
    coverage_miles: float,
) -> dict:
    """DISCOVER: look at parent+cell images, list anchor candidates."""
    client = _get_client()
    system_prompt = _load_prompt("discover_anchors_system.txt")
    user_template = _load_prompt("discover_anchors_user.txt")

    context_line = ""
    if parent_context:
        context_line = f"\nParent-cell context:\n{parent_context}\n"

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
                    _image_block(z15_image_path),
                    _image_block(z16_image_path),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


# --- RESOLVE_CONTINUATION ---


async def resolve_continuation(
    mosaic_image_path: str,
    anchor: dict,
    mosaic_width: int,
    mosaic_height: int,
    mosaic_rows: int,
    mosaic_cols: int,
    anchor_center: tuple[float, float],
) -> dict:
    """RESOLVE_CONTINUATION: does the anchor run off the current mosaic?"""
    client = _get_client()
    system_prompt = _load_prompt("resolve_continuation_system.txt")
    user_template = _load_prompt("resolve_continuation_user.txt")

    prior = anchor.get("continuation_edges") or {}
    prior_edges = ", ".join(
        f"{k}={bool(prior.get(k))}" for k in ("north", "south", "east", "west")
    )

    user_prompt = user_template.format(
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
        mosaic_width=mosaic_width,
        mosaic_height=mosaic_height,
        mosaic_rows=mosaic_rows,
        mosaic_cols=mosaic_cols,
        anchor_lat=f"{anchor_center[0]:.5f}",
        anchor_lon=f"{anchor_center[1]:.5f}",
        prior_edges=prior_edges,
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(mosaic_image_path),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


# --- IDENTIFY_ANCHOR (seeds + members + influence polygon) ---


async def identify_anchor(
    mosaic_image_path: str,
    anchor: dict,
    feedback_note: str = "",
) -> dict:
    """IDENTIFY_ANCHOR: click points for the anchor and its complex members,
    plus the influence-zone polygon.

    `feedback_note` is appended to the user prompt when the validator has
    rejected an earlier attempt and we're asking for a regeneration. It
    describes the specific seed issue to correct.
    """
    client = _get_client()
    system_prompt = _load_prompt("identify_anchor_system.txt")
    user_template = _load_prompt("identify_anchor_user.txt")

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
    )
    if feedback_note:
        user_prompt = (
            user_prompt
            + "\n\nNOTE: a previous attempt at these seeds failed validation. "
            + f"Issue: {feedback_note}. Please correct it in this response.\n"
        )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(mosaic_image_path),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


# --- IDENTIFY_SUBZONES (seeds for subzones only) ---


async def identify_subzones(
    mosaic_image_path: str,
    anchor: dict,
    feedback_note: str = "",
) -> dict:
    """IDENTIFY_SUBZONES: click points per subzone, constrained to v1 whitelist.

    Like identify_anchor, accepts an optional feedback_note for regeneration.
    """
    client = _get_client()
    system_prompt = _load_prompt("identify_subzones_system.txt")
    user_template = _load_prompt("identify_subzones_user.txt")

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
    )
    if feedback_note:
        user_prompt = (
            user_prompt
            + "\n\nNOTE: a previous attempt at these seeds failed validation. "
            + f"Issue: {feedback_note}. Please correct it in this response.\n"
        )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(mosaic_image_path),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed
