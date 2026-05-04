"""Prompt experimentation harness for the structure-phase prompts.

Loads a named prompt variant from `prompts_variants/<name>/`, runs it against
one test cell N times, and produces:
  - gridded input image (what Claude sees)
  - verification image per run (anchors outlined on the gridded image)
  - raw response per run (.md)
  - parsed JSON per run
  - per-cell summary across runs (consistency check)

Usage:
  python scripts/prompt_experiment.py --variant v2 --cell root-10-8 --repeats 3
  python scripts/prompt_experiment.py --variant v2 --cell both --repeats 3

Variant directory layout:
  prompts_variants/<variant>/
    discover_anchors_system.txt
    discover_anchors_user.txt

Any file missing from the variant dir falls back to the main `prompts/` dir,
so you only need to drop in the files you're changing.

Outputs go to:
  prompt_experiments/<timestamp>_<variant>/
    <cell_id>/
      gridded_input.png
      run_1/response.md, cells.json, verification.png
      run_2/...
      summary.json   (per-anchor stability across runs)
    diff_notes.md    (written by you when comparing variants)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from datetime import datetime
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
sys.path.insert(0, str(WORKTREE / "scripts"))

import anthropic  # noqa: E402

from _cells import CELLS  # noqa: E402
from readwater.pipeline.evidence import (  # noqa: E402
    build_evidence_table,
    format_evidence_for_prompt,
)
from readwater.pipeline.grid_overlay import (  # noqa: E402
    cells_to_bbox,
    draw_label_grid,
    grid_shape_for_image,
    row_label,
)

MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 6000

VARIANTS_ROOT = WORKTREE / "prompts_variants"
MAIN_PROMPTS = WORKTREE / "prompts"
EXPERIMENTS_ROOT = REPO_ROOT / "data" / "prompt_experiments"
EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)


# --- Prompt resolution ---


def load_variant_prompt(variant: str, filename: str) -> str:
    """Load a prompt file from the variant dir, falling back to main prompts."""
    variant_path = VARIANTS_ROOT / variant / filename
    if variant_path.exists():
        return variant_path.read_text(encoding="utf-8").strip()
    main_path = MAIN_PROMPTS / filename
    if main_path.exists():
        return main_path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"neither {variant_path} nor {main_path} exist")


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


def _extract_json(text: str) -> dict:
    matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())
    matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return json.loads(matches[-1].group(1).strip())
    return json.loads(text.strip())


def _auto_discover_mask_paths(cell_id: str) -> dict[str, str]:
    """Look for pre-generated masks for this cell at the standard output paths.

    Returns only layers whose mask file exists on disk, so missing layers
    degrade gracefully to "no evidence for that layer."
    """
    data_root = REPO_ROOT / "data" / "areas"
    candidates = {
        "water":    data_root / "rookery_bay_v2_naip"    / f"{cell_id}_water_mask.png",
        "channel":  data_root / "rookery_bay_v2_channels" / f"{cell_id}_channel_mask.png",
        "oyster":   data_root / "rookery_bay_v2_habitats" / f"{cell_id}_oyster_mask.png",
        "seagrass": data_root / "rookery_bay_v2_habitats" / f"{cell_id}_seagrass_mask.png",
    }
    found = {k: str(v) for k, v in candidates.items() if v.exists()}
    return found


async def run_once(
    z15_path: str,
    z16_gridded_path: str,
    cell_meta: dict,
    grid_rows: int,
    grid_cols: int,
    coverage_miles: float,
    variant: str,
    evidence_masks: dict[str, str] | None = None,
) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_prompt = load_variant_prompt(variant, "discover_anchors_system.txt")
    user_template = load_variant_prompt(variant, "discover_anchors_user.txt")

    tile_side_ft = int(round(coverage_miles * 5280))
    cell_side_ft = int(round(tile_side_ft / max(grid_rows, grid_cols)))
    small_feature_cells = max(1, round((500 / cell_side_ft) ** 2)) if cell_side_ft else 1
    large_feature_cells = max(1, round((1500 / cell_side_ft) ** 2)) if cell_side_ft else 1

    context_line = ""
    if cell_meta.get("parent_context"):
        context_line = f"\nParent-cell context:\n{cell_meta['parent_context']}\n"

    evidence_section = ""
    if evidence_masks:
        evidence = build_evidence_table(
            mask_paths=evidence_masks,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            image_size=(1280, 1280),
        )
        evidence_section = format_evidence_for_prompt(evidence)

    user_prompt = user_template.format(
        parent_context=context_line,
        center_lat=f"{cell_meta['cell_center'][0]:.4f}",
        center_lon=f"{cell_meta['cell_center'][1]:.4f}",
        coverage_miles=f"{coverage_miles:.2f}",
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        last_row=row_label(grid_rows - 1),
        evidence_section=evidence_section,
        tile_side_ft=tile_side_ft,
        cell_side_ft=cell_side_ft,
        small_feature_cells=small_feature_cells,
        large_feature_cells=large_feature_cells,
    )

    # Retry with exponential backoff on transient API errors (overload, rate
    # limit, brief network failures). Discovery is a single Claude call and
    # losing a run midway through an 8-cell sweep means redoing everything,
    # so we give it a few tries.
    max_retries = 5
    delay = 15.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
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
            break
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_err = e
            if attempt + 1 == max_retries:
                raise
            print(f"  [retry] attempt {attempt + 1} failed ({type(e).__name__}: "
                  f"{e}); sleeping {delay:.0f}s and retrying...")
            await asyncio.sleep(delay)
            delay *= 2
    else:
        raise RuntimeError(f"exhausted retries: {last_err}")
    raw = resp.content[0].text
    parsed = _extract_json(raw)
    parsed["_raw"] = raw
    parsed["_user_prompt"] = user_prompt
    return parsed


# --- Verification image ---


PALETTE = [
    (255, 215, 0),
    (0, 191, 255),
    (255, 69, 0),
    (50, 205, 50),
    (238, 130, 238),
    (255, 165, 0),
]


def draw_verification(
    gridded_path: str, anchors: list[dict], grid_rows: int, grid_cols: int,
    image_size: tuple[int, int], out_path: Path,
) -> None:
    img = Image.open(gridded_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, a in enumerate(anchors):
        color = PALETTE[i % len(PALETTE)]
        bbox = cells_to_bbox(a.get("cells", []), grid_rows, grid_cols, image_size)
        if bbox is None:
            continue
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=5)
        label = f"{a.get('anchor_id', '?')} {a.get('structure_type', '')}"
        draw.rectangle([x, y, x + min(280, w), y + 30], fill=(0, 0, 0, 210))
        draw.text((x + 6, y + 4), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# --- Consistency summary ---


def consistency_summary(runs: list[dict]) -> dict:
    """For each anchor (keyed by structure_type), report cell stability across runs.

    Since Claude picks different IDs per run, we group anchors across runs by
    structure_type and cell-overlap. For each group, report: n_runs it appeared
    in, intersection of cells, union of cells, IoU across runs.
    """
    # Flatten: (run_idx, anchor)
    all_anchors = []
    for ri, resp in enumerate(runs):
        for a in resp.get("anchors", []):
            all_anchors.append((ri, a))

    # Greedy cluster: match anchors across runs by structure_type + cell overlap.
    clusters: list[list[tuple[int, dict]]] = []
    for ri, a in all_anchors:
        a_cells = set(a.get("cells", []))
        placed = False
        for c in clusters:
            first_run, first_a = c[0]
            if first_a.get("structure_type") != a.get("structure_type"):
                continue
            first_cells = set(first_a.get("cells", []))
            if not a_cells and not first_cells:
                continue
            inter = len(a_cells & first_cells)
            union = len(a_cells | first_cells)
            iou = inter / union if union else 0
            if iou >= 0.25:
                c.append((ri, a))
                placed = True
                break
        if not placed:
            clusters.append([(ri, a)])

    summary_groups = []
    for c in clusters:
        cells_per_run = [set(x[1].get("cells", [])) for x in c]
        rationales = [x[1].get("rationale", "") for x in c]
        inter = set.intersection(*cells_per_run) if cells_per_run else set()
        union = set.union(*cells_per_run) if cells_per_run else set()
        runs_present = sorted({x[0] for x in c})
        summary_groups.append({
            "structure_type": c[0][1].get("structure_type"),
            "runs_present_in": runs_present,
            "n_runs": len(runs_present),
            "total_runs": len(runs),
            "cells_intersection": sorted(inter),
            "cells_union": sorted(union),
            "stability": len(inter) / len(union) if union else 0.0,
            "per_run_cells": [sorted(s) for s in cells_per_run],
            "per_run_rationale": rationales,
        })

    summary_groups.sort(key=lambda g: (-g["n_runs"], -len(g["cells_intersection"])))
    return {
        "n_runs": len(runs),
        "n_clusters": len(summary_groups),
        "anchors_per_run": [len(r.get("anchors", [])) for r in runs],
        "clusters": summary_groups,
    }


# --- Main ---


async def run_for_cell(
    variant: str,
    cell_id: str,
    repeats: int,
    exp_dir: Path,
    use_evidence: bool = False,
) -> None:
    cell_meta = CELLS[cell_id]
    z16 = str(cell_meta["z16"])
    z15 = str(cell_meta["z15"])

    cell_dir = exp_dir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Determine grid shape from the z16 image (square, 8x8 default).
    img = Image.open(z16)
    grid_rows, grid_cols = grid_shape_for_image(img.size, short_axis_cells=8)

    # Draw grid once; all runs share the same input.
    gridded_path = cell_dir / "gridded_input.png"
    draw_label_grid(z16, grid_rows, grid_cols, str(gridded_path))

    evidence_masks: dict[str, str] | None = None
    if use_evidence:
        evidence_masks = _auto_discover_mask_paths(cell_id)
        if not evidence_masks:
            print(f"WARNING [{cell_id}]: --with-evidence was set but no masks "
                  f"were found at the standard paths. Runs will proceed without evidence.")
            evidence_masks = None
        else:
            print(f"[{cell_id}] evidence layers found: {sorted(evidence_masks.keys())}")

    print(f"\n=== {cell_id}  variant={variant}  repeats={repeats}  "
          f"grid={grid_rows}x{grid_cols}  evidence={'on' if evidence_masks else 'off'} ===")
    print(f"gridded input: {gridded_path}")

    runs: list[dict] = []
    for i in range(repeats):
        print(f"\n--- run {i + 1}/{repeats} ---")
        resp = await run_once(
            z15, str(gridded_path), cell_meta,
            grid_rows, grid_cols,
            coverage_miles=0.37,
            variant=variant,
            evidence_masks=evidence_masks,
        )
        raw = resp.pop("_raw", "")
        user_prompt = resp.pop("_user_prompt", "")

        run_dir = cell_dir / f"run_{i + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "response.md").write_text(raw, encoding="utf-8")
        (run_dir / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
        (run_dir / "cells.json").write_text(
            json.dumps(resp, indent=2), encoding="utf-8",
        )
        anchors = resp.get("anchors", [])
        draw_verification(
            str(gridded_path), anchors, grid_rows, grid_cols,
            img.size, run_dir / "verification.png",
        )
        print(f"  anchors: {len(anchors)}")
        for a in anchors:
            cells = a.get("cells", [])
            print(f"    {a.get('anchor_id', '?')}  {a.get('structure_type', '?'):15s}  "
                  f"scale={a.get('scale', '?'):5s}  conf={a.get('confidence', '?'):.2f}  "
                  f"cells={cells}")
        runs.append(resp)

    # Summary
    summary = consistency_summary(runs)
    (cell_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    print(f"\n--- consistency summary for {cell_id} ---")
    print(f"anchors per run: {summary['anchors_per_run']}")
    print(f"{summary['n_clusters']} distinct anchor clusters across {summary['n_runs']} runs")
    for g in summary["clusters"]:
        print(
            f"  {g['structure_type']:15s}  present in {g['n_runs']}/{g['total_runs']} runs  "
            f"stability={g['stability']:.2f}  "
            f"intersection={g['cells_intersection']}"
        )
    print(f"\nsummary: {cell_dir / 'summary.json'}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        help="variant name under prompts_variants/ (e.g. 'v2')")
    parser.add_argument("--cell", default="root-10-8",
                        choices=list(CELLS.keys()) + ["all", "both"],
                        help="Single cell id, 'all' for every cell in "
                             "_cells.CELLS, or 'both' (legacy alias for the "
                             "two original cells root-10-8 + root-11-5).")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-evidence", action="store_true",
                        help="auto-discover pre-generated water/channel/oyster/seagrass "
                             "masks for each cell and inject per-cell evidence into the "
                             "discovery prompt.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"{args.variant}{'_evid' if args.with_evidence else ''}"
    exp_dir = EXPERIMENTS_ROOT / f"{ts}_{suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    if args.cell == "all":
        cells = list(CELLS.keys())
    elif args.cell == "both":
        cells = ["root-10-8", "root-11-5"]
    else:
        cells = [args.cell]
    for cid in cells:
        await run_for_cell(
            args.variant, cid, args.repeats, exp_dir,
            use_evidence=args.with_evidence,
        )

    print(f"\nAll done. Browse outputs in: {exp_dir}")


if __name__ == "__main__":
    asyncio.run(main())
