"""Thin Claude-vision wrappers for the four structure-phase prompts.

These mirror the style of `api/claude_vision.py` but keep the structure-phase
prompt invocations isolated in one file. All functions:
  - load the relevant system/user prompt templates from prompts/,
  - format the user template,
  - call Claude with one or two images,
  - return the parsed JSON plus the raw response text under `raw_response`.
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

# Claude caps base64 images at 5 MB decoded. JPEG re-encoding is used for all
# uploads because the stitched mosaic PNGs routinely exceed this as PNG. A
# fallback resize catches the rare very-large image.
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 - 64 * 1024  # leave a little headroom
_UPLOAD_JPEG_QUALITY = 85
_UPLOAD_MAX_DIM = 1800


def _image_block(path: str | Path) -> dict:
    """Re-encode an image as JPEG (with an optional downscale) for Claude upload."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Cap longest dimension defensively; the agent already caps mosaic canvases
    # via MAX_MOSAIC_DIM, but z15/z16 source PNGs and any future inputs may be
    # larger.
    longest = max(img.size)
    if longest > _UPLOAD_MAX_DIM:
        ratio = _UPLOAD_MAX_DIM / longest
        new_size = (int(round(img.size[0] * ratio)), int(round(img.size[1] * ratio)))
        img = img.resize(new_size, Image.LANCZOS)

    quality = _UPLOAD_JPEG_QUALITY
    for _ in range(4):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= _MAX_UPLOAD_BYTES:
            break
        # Still too big -> drop quality or downscale further
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
    prior_edges = ", ".join(f"{k}={bool(prior.get(k))}" for k in ("north", "south", "east", "west"))

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


async def model_influence(
    mosaic_image_path: str,
    anchor: dict,
    mosaic_width: int,
    mosaic_height: int,
) -> dict:
    """MODEL_INFLUENCE: produce anchor/complex/influence polygons on the mosaic."""
    client = _get_client()
    system_prompt = _load_prompt("model_influence_system.txt")
    user_template = _load_prompt("model_influence_user.txt")

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
        mosaic_width=mosaic_width,
        mosaic_height=mosaic_height,
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


async def define_subzones(
    mosaic_image_path: str,
    anchor: dict,
    mosaic_width: int,
    mosaic_height: int,
) -> dict:
    """DEFINE_SUBZONES: compact fishable subzones within the influence zone."""
    client = _get_client()
    system_prompt = _load_prompt("define_subzones_system.txt")
    user_template = _load_prompt("define_subzones_user.txt")

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
        mosaic_width=mosaic_width,
        mosaic_height=mosaic_height,
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
