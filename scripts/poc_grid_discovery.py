"""POC: grid-overlay discovery on the zoom-16 cell image.

Overlays an 8x8 labeled grid (A1..H8, rows top->bottom, cols left->right) on
the zoom-16 cell image, sends it + the zoom-15 context image to Claude with
a cell-label discovery prompt, and prints what Claude picks.

Saves the overlaid input image next to the originals so it can be inspected
visually. No code in the main pipeline is changed by this script.

Usage:
  python scripts/poc_grid_discovery.py [--cell root-10-8|root-11-5]
  python scripts/poc_grid_discovery.py --grid 16          # use 16x16 instead
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import string
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

import anthropic  # noqa: E402

MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 4096

DATA_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_grid_poc"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CELLS = {
    "root-10-8": {
        "z16": DATA_ROOT / "z0_10_8.png",
        "z15": DATA_ROOT / "z0_10_8_context_z15.png",
        "cell_center": (26.011172, -81.753546),
        "parent_context": (
            "Rookery Bay, SW Florida. Parent zoom-14 cell shows mangrove-lined "
            "estuarine shoreline with tidal cuts and shallow basins."
        ),
    },
    "root-11-5": {
        "z16": DATA_ROOT / "z0_11_5.png",
        "z15": DATA_ROOT / "z0_11_5_context_z15.png",
        "cell_center": (26.011172, -81.739780),
        "parent_context": (
            "Rookery Bay, SW Florida. Parent zoom-14 cell shows interior bay "
            "water with mangrove islands and connecting channels."
        ),
    },
}


# --- Grid rendering ---


def _row_label(idx: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, etc. Handles up to 32 rows."""
    if idx < 26:
        return string.ascii_uppercase[idx]
    # Two-letter for big grids.
    first = (idx // 26) - 1
    second = idx % 26
    return string.ascii_uppercase[first] + string.ascii_uppercase[second]


def draw_label_grid(
    image_path: str,
    grid: int,
    out_path: str,
) -> str:
    """Draw a `grid` x `grid` overlay on the image with A1..<row><col> labels.

    Rows are A..Z top to bottom; columns are 1..grid left to right.
    """
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cell_w = w / grid
    cell_h = h / grid

    # Grid lines: thin white with 1 px black shadow
    for i in range(1, grid):
        x = int(i * cell_w)
        y = int(i * cell_h)
        draw.line([(x + 1, 0), (x + 1, h)], fill=(0, 0, 0, 200), width=1)
        draw.line([(0, y + 1), (w, y + 1)], fill=(0, 0, 0, 200), width=1)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 230), width=2)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 230), width=2)

    # Labels: row letter + column number, centered in each cell
    font_size = max(10, int(min(cell_w, cell_h) * 0.32))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size,
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for row in range(grid):
        for col in range(grid):
            label = f"{_row_label(row)}{col + 1}"
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            # Black outline (9 offsets) + white fill
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, fill="black", font=font)
            draw.text((tx, ty), label, fill="white", font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# --- Claude call ---


def _image_block_jpeg(path: str, max_dim: int = 1800, quality: int = 85) -> dict:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        r = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * r), int(img.size[1] * r)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
        },
    }


SYSTEM_PROMPT = """You are an expert inshore saltwater fishing guide reviewing satellite imagery.
You will see a zoom-15 parent context image and a zoom-16 cell image with a labeled grid overlay.
The grid labels work like spreadsheet coordinates: row letters (A = topmost, B = next row down, etc.) paired with column numbers (1 = leftmost, 2 = next column right, etc.). So A1 is the top-left cell, A8 is the top-right cell in an 8x8 grid, H1 is the bottom-left, H8 is the bottom-right.

Your job is to identify the major ANCHOR STRUCTURES in the zoom-16 cell image — the handful of features that most strongly organize where fish hold and feed (drains, points, coves, oyster bars, troughs, creek mouths, island-edge systems, current splits around islands).

For EACH anchor, return:
- structure_type: drain | point | cove | oyster_bar | island_edge | trough | creek_mouth | pass | shoreline_bend | current_split | mangrove_peninsula | shoreline_cut
- scale: major | minor
- rationale: one sentence on what you see
- confidence: 0.0 to 1.0
- cells: the list of grid-cell labels that contain the anchor's footprint. Include every cell the feature visibly occupies. At least 1 cell, typically 2-8.

Rules:
- Return at most 6 anchors. Fewer, higher-quality is better.
- Only pick cells that actually contain visible parts of the feature. Do NOT pad.
- Verify cell labels before returning. Read the label in the grid image directly — do not guess row/column positions.
"""

USER_TEMPLATE = """Image 1: zoom-15 parent context (~0.75 mi per side).
Image 2: zoom-16 cell with a {grid}x{grid} labeled grid overlay (~0.37 mi per side).

Parent-cell context: {parent_context}
Zoom-16 cell center: ({center_lat}, {center_lon})

Identify the anchor structures and their grid cells.

```json
{{
  "summary": "one-line description of the tile as a fishing picture",
  "anchors": [
    {{
      "anchor_id": "a1",
      "structure_type": "drain",
      "scale": "major",
      "rationale": "narrow tidal cut draining a shallow back bay",
      "confidence": 0.85,
      "cells": ["D4", "D5", "E4", "E5"]
    }}
  ]
}}
```
"""


def _extract_json(text: str) -> dict:
    matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())
    matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())
    return json.loads(text.strip())


async def call_claude(z15_path: str, z16_gridded_path: str, cell: dict, grid: int) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = USER_TEMPLATE.format(
        grid=grid,
        parent_context=cell["parent_context"],
        center_lat=f"{cell['cell_center'][0]:.4f}",
        center_lon=f"{cell['cell_center'][1]:.4f}",
    )
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block_jpeg(z15_path),
                    _image_block_jpeg(z16_gridded_path),
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    raw = resp.content[0].text
    parsed = _extract_json(raw)
    parsed["_raw"] = raw
    return parsed


# --- Cell → pixel bbox (for verification drawing) ---


def cells_to_bbox(cell_labels: list[str], grid: int, image_size: int = 1280) -> tuple[int, int, int, int]:
    """Union of the given cell labels as an (x, y, w, h) pixel bbox."""
    cw = image_size / grid
    ch = image_size / grid
    xs, ys = [], []
    for label in cell_labels:
        if len(label) < 2:
            continue
        # Parse row letters and column number
        m = re.match(r"^([A-Z]+)(\d+)$", label.upper())
        if not m:
            continue
        rowpart, colpart = m.group(1), m.group(2)
        # Row index: single letter A..Z → 0..25; two letters AA..AZ → 26..51 etc.
        if len(rowpart) == 1:
            row = ord(rowpart) - ord("A")
        else:
            row = (ord(rowpart[0]) - ord("A") + 1) * 26 + (ord(rowpart[1]) - ord("A"))
        col = int(colpart) - 1
        xs.append(col * cw)
        xs.append((col + 1) * cw)
        ys.append(row * ch)
        ys.append((row + 1) * ch)
    if not xs:
        return (0, 0, 0, 0)
    x = int(min(xs))
    y = int(min(ys))
    w = int(max(xs) - x)
    h = int(max(ys) - y)
    return (x, y, w, h)


def draw_verification_image(
    z16_gridded_path: str,
    anchors: list[dict],
    grid: int,
    out_path: str,
    image_size: int = 1280,
) -> str:
    """Draw colored outlines around the cells Claude picked for each anchor."""
    img = Image.open(z16_gridded_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()

    palette = [
        (255, 215, 0),   # gold
        (0, 191, 255),   # sky blue
        (255, 69, 0),    # red-orange
        (50, 205, 50),   # lime
        (238, 130, 238), # violet
        (255, 165, 0),   # orange
    ]
    for i, a in enumerate(anchors):
        color = palette[i % len(palette)]
        x, y, w, h = cells_to_bbox(a.get("cells", []), grid, image_size)
        if w == 0 or h == 0:
            continue
        # Bold outline
        draw.rectangle([x, y, x + w, y + h], outline=color, width=5)
        # Label band
        label = f"{a.get('anchor_id', '?')} {a.get('structure_type', '')}"
        draw.rectangle([x, y, x + min(300, w), y + 30], fill=(0, 0, 0, 200))
        draw.text((x + 6, y + 4), label, fill=color, font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# --- Main ---


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=list(CELLS.keys()), default="root-10-8")
    parser.add_argument("--grid", type=int, default=8, choices=[8, 16])
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    cell = CELLS[args.cell]
    print(f"=== Grid-overlay discovery POC: {args.cell}  grid={args.grid}x{args.grid} ===")

    # Step 1: draw the grid overlay on the zoom-16 cell image
    gridded_path = OUT_ROOT / f"{args.cell}_z16_grid{args.grid}.png"
    draw_label_grid(str(cell["z16"]), args.grid, str(gridded_path))
    print(f"gridded input saved to: {gridded_path}")

    # Step 2: call Claude
    print("calling Claude...")
    resp = await call_claude(
        str(cell["z15"]), str(gridded_path), cell, args.grid,
    )
    raw = resp.pop("_raw", "")

    # Step 3: print response
    print()
    print("--- Claude response ---")
    print("summary:", resp.get("summary", ""))
    print()
    anchors = resp.get("anchors", [])
    for a in anchors:
        cells = a.get("cells", [])
        print(f"  {a.get('anchor_id', '?')}  {a.get('structure_type', '?'):15s}  "
              f"scale={a.get('scale', '?'):5s}  conf={a.get('confidence', '?')}")
        print(f"    cells: {cells}")
        print(f"    rationale: {a.get('rationale', '')}")

    # Step 4: draw verification image
    verify_path = OUT_ROOT / f"{args.cell}_z16_grid{args.grid}_verification.png"
    draw_verification_image(
        str(gridded_path), anchors, args.grid, str(verify_path),
    )
    print()
    print(f"verification image saved to: {verify_path}")

    # Step 5: save raw response
    md_path = OUT_ROOT / f"{args.cell}_z16_grid{args.grid}.md"
    md_path.write_text(raw, encoding="utf-8")
    print(f"raw response saved to: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
