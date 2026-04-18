"""POC: grid-overlay identify on an existing zoom-18 mosaic.

Overlays a 16x16 labeled grid on a stitched zoom-18 mosaic already on disk
and asks Claude to identify:
  - the anchor's core cells
  - local-complex member features (name + feature_type + cells)
  - up to 4 subzones from the v1 whitelist (each with type + cells)

No new API fetches; no influence-zone polygon this pass (already known
Claude can sketch that).

Usage:
  python scripts/poc_grid_identify.py
  python scripts/poc_grid_identify.py --grid 12
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
MAX_TOKENS = 6000

A2_MOSAIC = (
    REPO_ROOT
    / "data/areas/rookery_bay_v2_structure_test/images/cells/root-10-8/structures/a2/mosaic.png"
)
OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_grid_poc"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# --- Grid rendering ---


def _row_label(idx: int) -> str:
    if idx < 26:
        return string.ascii_uppercase[idx]
    first = (idx // 26) - 1
    second = idx % 26
    return string.ascii_uppercase[first] + string.ascii_uppercase[second]


def draw_label_grid(image_path: str, grid: int, out_path: str) -> str:
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cell_w = w / grid
    cell_h = h / grid

    for i in range(1, grid):
        x = int(i * cell_w)
        y = int(i * cell_h)
        draw.line([(x + 1, 0), (x + 1, h)], fill=(0, 0, 0, 200), width=1)
        draw.line([(0, y + 1), (w, y + 1)], fill=(0, 0, 0, 200), width=1)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 230), width=2)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 230), width=2)

    font_size = max(9, int(min(cell_w, cell_h) * 0.30))
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


SYSTEM_PROMPT = """You are an expert inshore saltwater fishing guide looking at a detailed zoom-18 satellite mosaic with a {grid}x{grid} labeled grid overlay.

Grid labels read like spreadsheet coordinates: row letters (A = topmost, B = next row down, ..., P = bottommost in 16x16) paired with column numbers (1 = leftmost, 2 = next column right, ..., 16 = rightmost). So A1 is the top-left cell, A{grid} is the top-right, the bottom-right is the last row letter + {grid}.

An ANCHOR structure has already been discovered in this mosaic:
  type: {structure_type}
  rationale: {rationale}

Your job is to identify, by grid cell, three kinds of things:

1. ANCHOR CORE
   The cells that contain the actual visible anchor feature. Keep tight — only cells where the feature is clearly present. Typically 2-10 cells.

2. LOCAL COMPLEX MEMBER FEATURES
   Nearby visible features that function with the anchor (flanking points, receiving basins, adjacent bar segments, mangrove fingers, etc.). Each member has:
     - name: short human label ("left flanking point", "receiving basin")
     - feature_type: point | basin | bar | shoreline | channel | pocket | spit | mangrove_finger
     - cells: the grid cells that contain this member (typically 1-6)
   0 to 5 members. Only include ones that genuinely fish together with the anchor.

3. FISHABLE SUBZONES
   Compact fishing targets within the anchor's influence. Restricted to the v1 whitelist:
     drain_throat | point_tip | oyster_bar_edge | pocket_mouth | island_tip_seam
   Each subzone has:
     - subzone_type: exactly one of the five above
     - cells: typically 1-3 (subzones are small, stable targets)
     - relative_priority: 0.0 to 1.0 (which to fish first under good conditions)
     - reasoning_summary: one line
     - confidence: 0.0 to 1.0
   0 to 4 subzones.

Rules:
- Only pick cells that actually contain visible parts of the feature. Do NOT pad.
- Verify each cell label by reading it in the image directly. Do not guess.
- Subzone types must be exactly from the whitelist above.
- Keep the anchor core cells tight: fewer, better.
"""

USER_TEMPLATE = """Image: stitched zoom-18 mosaic with a {grid}x{grid} labeled grid overlay.

Identify the anchor core cells, the local-complex member features with their cells, and the subzones with their cells.

```json
{{
  "anchor_cells": ["F7", "F8", "G7", "G8"],
  "anchor_notes": "brief note on what you see and where",
  "local_complex": {{
    "members": [
      {{
        "name": "left flanking point",
        "feature_type": "point",
        "cells": ["F5", "G5"]
      }}
    ],
    "relationship_summary": "one-line description of how these fish together"
  }},
  "subzones": [
    {{
      "subzone_id": "s1",
      "subzone_type": "drain_throat",
      "cells": ["F7"],
      "relative_priority": 1.0,
      "reasoning_summary": "tightest funnel",
      "confidence": 0.85
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


async def call_claude(gridded_path: str, grid: int, structure_type: str, rationale: str) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = SYSTEM_PROMPT.format(
        grid=grid, structure_type=structure_type, rationale=rationale,
    )
    user = USER_TEMPLATE.format(grid=grid)
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    _image_block_jpeg(gridded_path),
                    {"type": "text", "text": user},
                ],
            }
        ],
    )
    raw = resp.content[0].text
    parsed = _extract_json(raw)
    parsed["_raw"] = raw
    return parsed


# --- Cells → pixel bbox for verification drawing ---


def cells_to_bbox(cell_labels, grid: int, image_w: int, image_h: int):
    cw = image_w / grid
    ch = image_h / grid
    xs, ys = [], []
    for label in cell_labels or []:
        m = re.match(r"^([A-Z]+)(\d+)$", label.upper())
        if not m:
            continue
        rowpart, colpart = m.group(1), m.group(2)
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


def _outline_cells(draw, cell_labels, grid, image_w, image_h, color, width=5):
    cw = image_w / grid
    ch = image_h / grid
    for label in cell_labels or []:
        m = re.match(r"^([A-Z]+)(\d+)$", label.upper())
        if not m:
            continue
        rowpart, colpart = m.group(1), m.group(2)
        if len(rowpart) == 1:
            row = ord(rowpart) - ord("A")
        else:
            row = (ord(rowpart[0]) - ord("A") + 1) * 26 + (ord(rowpart[1]) - ord("A"))
        col = int(colpart) - 1
        x0 = int(col * cw)
        y0 = int(row * ch)
        x1 = int((col + 1) * cw)
        y1 = int((row + 1) * ch)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)


def draw_verification(
    gridded_path: str,
    resp: dict,
    grid: int,
    out_path: str,
) -> str:
    img = Image.open(gridded_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    try:
        font_big = ImageFont.truetype("arial.ttf", 26)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except (OSError, IOError):
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    anchor_color = (255, 215, 0)      # gold
    member_palette = [
        (0, 191, 255),   # sky blue
        (30, 144, 255),  # dodger blue
        (100, 149, 237), # cornflower
        (65, 105, 225),  # royal
        (0, 0, 205),     # medium blue
    ]
    subzone_color = (255, 69, 0)      # orange-red

    # Anchor core (wide solid outline)
    anchor_cells = resp.get("anchor_cells", [])
    _outline_cells(draw, anchor_cells, grid, w, h, anchor_color, width=6)
    # Anchor label
    if anchor_cells:
        x, y, aw, ah = cells_to_bbox(anchor_cells, grid, w, h)
        draw.rectangle([x, y, x + min(240, aw), y + 34], fill=(0, 0, 0, 200))
        draw.text((x + 6, y + 4), "ANCHOR", fill=anchor_color, font=font_big)

    # Members
    for idx, m in enumerate(resp.get("local_complex", {}).get("members", [])):
        color = member_palette[idx % len(member_palette)]
        cells = m.get("cells", [])
        _outline_cells(draw, cells, grid, w, h, color, width=5)
        if cells:
            x, y, mw, mh = cells_to_bbox(cells, grid, w, h)
            label = f"{m.get('name', '?')} ({m.get('feature_type', '?')})"
            draw.rectangle([x, y, x + min(360, mw), y + 26], fill=(0, 0, 0, 200))
            draw.text((x + 6, y + 3), label, fill=color, font=font_small)

    # Subzones
    for idx, s in enumerate(resp.get("subzones", [])):
        cells = s.get("cells", [])
        _outline_cells(draw, cells, grid, w, h, subzone_color, width=4)
        if cells:
            x, y, sw, sh = cells_to_bbox(cells, grid, w, h)
            label = f"{s.get('subzone_id', '?')} {s.get('subzone_type', '?')}"
            draw.rectangle([x, y, x + min(320, sw), y + 24], fill=(0, 0, 0, 200))
            draw.text((x + 6, y + 3), label, fill=subzone_color, font=font_small)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# --- Main ---


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=16, choices=[8, 12, 16])
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    if not A2_MOSAIC.exists():
        raise SystemExit(f"a2 mosaic not found at {A2_MOSAIC}")

    # Use the actual stored anchor metadata for context
    result_json = A2_MOSAIC.parent / "result.json"
    meta = json.loads(result_json.read_text())
    structure_type = meta["anchor"]["structure_type"]
    rationale = meta["anchor"]["rationale"]

    print(f"=== Grid-overlay identify POC: a2 ({structure_type})  grid={args.grid}x{args.grid} ===")
    print(f"rationale: {rationale}")
    print()

    gridded_path = OUT_ROOT / f"a2_z18_grid{args.grid}.png"
    draw_label_grid(str(A2_MOSAIC), args.grid, str(gridded_path))
    print(f"gridded input: {gridded_path}")

    print("calling Claude...")
    resp = await call_claude(str(gridded_path), args.grid, structure_type, rationale)
    raw = resp.pop("_raw", "")

    print()
    print("--- Claude response ---")
    print(f"anchor_cells: {resp.get('anchor_cells', [])}")
    print(f"anchor_notes: {resp.get('anchor_notes', '')}")
    print()
    members = resp.get("local_complex", {}).get("members", [])
    print(f"members: {len(members)}")
    for m in members:
        print(f"  {m.get('name', '?')} ({m.get('feature_type', '?')}): {m.get('cells', [])}")
    print(f"  relationship_summary: {resp.get('local_complex', {}).get('relationship_summary', '')}")
    print()
    subs = resp.get("subzones", [])
    print(f"subzones: {len(subs)}")
    for s in subs:
        print(f"  {s.get('subzone_id', '?')} ({s.get('subzone_type', '?')}): "
              f"cells={s.get('cells', [])} prio={s.get('relative_priority', '?')}")
        print(f"    {s.get('reasoning_summary', '')}")

    verify_path = OUT_ROOT / f"a2_z18_grid{args.grid}_verification.png"
    draw_verification(str(gridded_path), resp, args.grid, str(verify_path))
    print()
    print(f"verification: {verify_path}")
    md_path = OUT_ROOT / f"a2_z18_grid{args.grid}.md"
    md_path.write_text(raw, encoding="utf-8")
    print(f"raw response: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
