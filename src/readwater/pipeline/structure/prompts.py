"""Claude-vision wrappers for the structure-phase prompts (grid-cell edition).

Phase 1.5 prompts:
  discover_anchors      — at cell level, enumerate candidate anchors with
                          grid cells and continuation edges
  resolve_continuation  — on a mosaic, decide whether to expand tiles
  identify_anchor       — anchor cells + member cells + influence cells
  identify_subzones     — cells per subzone (v1 whitelist)

The discovery and identify wrappers here render a grid overlay on the image
(using pipeline.structure.grid_overlay) BEFORE sending to Claude, so the
image Claude sees always has labeled cells to read. The raw image path is
kept around in the agent for rendering the annotated output.
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
from readwater.pipeline.structure.grid_overlay import (
    draw_label_grid,
    grid_shape_for_image,
    row_label,
)

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 - 64 * 1024
_UPLOAD_JPEG_QUALITY = 85
_UPLOAD_MAX_DIM = 1800

# Grid sizing knobs. 8 divisions on the short axis keeps cells large enough
# for Claude to label reliably (from POC) while still being fine enough at
# z18 (~55 m per cell) for fishing-spot scale.
DEFAULT_DISCOVER_SHORT = 8
DEFAULT_IDENTIFY_SHORT = 8


def _image_block(path: str | Path) -> dict:
    """JPEG-encode with resize-if-too-large for Claude upload."""
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


def _gridded_image_path(
    src_image_path: str,
    short_axis_cells: int,
    out_dir: str | Path,
    suffix: str,
) -> tuple[str, int, int]:
    """Draw a grid overlay on src image; return (gridded_path, rows, cols)."""
    img = Image.open(src_image_path)
    rows, cols = grid_shape_for_image(img.size, short_axis_cells=short_axis_cells)
    out = Path(out_dir) / f"{Path(src_image_path).stem}__grid{rows}x{cols}_{suffix}.png"
    draw_label_grid(src_image_path, rows, cols, str(out))
    return (str(out), rows, cols)


# --- DISCOVER ---


async def discover_anchors(
    z15_image_path: str,
    z16_image_path: str,
    parent_context: str,
    center: tuple[float, float],
    coverage_miles: float,
    grid_out_dir: str | Path,
    short_axis_cells: int = DEFAULT_DISCOVER_SHORT,
) -> dict:
    """DISCOVER: parent z15 + gridded z16 → anchor candidates with cell lists."""
    client = _get_client()
    system_prompt = _load_prompt("discover_anchors_system.txt")
    user_template = _load_prompt("discover_anchors_user.txt")

    gridded_z16, rows, cols = _gridded_image_path(
        z16_image_path, short_axis_cells, grid_out_dir, "discover",
    )

    context_line = ""
    if parent_context:
        context_line = f"\nParent-cell context:\n{parent_context}\n"

    user_prompt = user_template.format(
        parent_context=context_line,
        center_lat=f"{center[0]:.4f}",
        center_lon=f"{center[1]:.4f}",
        coverage_miles=f"{coverage_miles:.2f}",
        grid_rows=rows,
        grid_cols=cols,
        last_row=row_label(rows - 1),
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
                    _image_block(gridded_z16),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    parsed["_grid_rows"] = rows
    parsed["_grid_cols"] = cols
    parsed["_gridded_image_path"] = gridded_z16
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


# --- IDENTIFY_ANCHOR ---


async def identify_anchor(
    mosaic_image_path: str,
    anchor: dict,
    grid_out_dir: str | Path,
    short_axis_cells: int = DEFAULT_IDENTIFY_SHORT,
    feedback_note: str = "",
) -> dict:
    """IDENTIFY_ANCHOR: gridded mosaic → anchor/member/influence cell lists."""
    client = _get_client()
    system_prompt = _load_prompt("identify_anchor_system.txt")
    user_template = _load_prompt("identify_anchor_user.txt")

    gridded_mosaic, rows, cols = _gridded_image_path(
        mosaic_image_path, short_axis_cells, grid_out_dir, "identify",
    )

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
        grid_rows=rows,
        grid_cols=cols,
        last_row=row_label(rows - 1),
    )
    if feedback_note:
        user_prompt += (
            f"\n\nNOTE: a previous attempt at these cells failed validation. "
            f"Issue: {feedback_note}. Please correct it in this response.\n"
        )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(gridded_mosaic),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    parsed["_grid_rows"] = rows
    parsed["_grid_cols"] = cols
    parsed["_gridded_image_path"] = gridded_mosaic
    return parsed


# --- IDENTIFY_SUBZONES ---


async def identify_subzones(
    mosaic_image_path: str,
    anchor: dict,
    grid_out_dir: str | Path,
    short_axis_cells: int = DEFAULT_IDENTIFY_SHORT,
    feedback_note: str = "",
) -> dict:
    """IDENTIFY_SUBZONES: gridded mosaic → cell lists per subzone (v1 whitelist)."""
    client = _get_client()
    system_prompt = _load_prompt("identify_subzones_system.txt")
    user_template = _load_prompt("identify_subzones_user.txt")

    gridded_mosaic, rows, cols = _gridded_image_path(
        mosaic_image_path, short_axis_cells, grid_out_dir, "subzones",
    )

    user_prompt = user_template.format(
        anchor_id=anchor.get("anchor_id", "a1"),
        structure_type=anchor.get("structure_type", "unknown"),
        scale=anchor.get("scale", "minor"),
        rationale=anchor.get("rationale", ""),
        grid_rows=rows,
        grid_cols=cols,
        last_row=row_label(rows - 1),
    )
    if feedback_note:
        user_prompt += (
            f"\n\nNOTE: a previous attempt at these cells failed validation. "
            f"Issue: {feedback_note}. Please correct it in this response.\n"
        )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block(gridded_mosaic),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw_text = response.content[0].text
    parsed = _extract_json_from_response(raw_text)
    parsed["raw_response"] = raw_text
    parsed["_grid_rows"] = rows
    parsed["_grid_cols"] = cols
    parsed["_gridded_image_path"] = gridded_mosaic
    return parsed
