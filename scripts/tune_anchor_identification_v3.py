"""Iteration harness for tuning anchor_identification_v3 prompts.

v3 ships in two grid-mode variants:
  - nogrid: prompts/anchor_identification_v3_nogrid_{system,user}.txt
  - grid:   prompts/anchor_identification_v3_grid_{system,user}.txt

Drives the chosen variant against a single tuning cell (root-10-8), calls
Claude with the 4-image handoff + context bundle, parses the JSON output,
and scores it against a hand-labeled ground truth for that cell.

Intended workflow:
  1. Edit one or both variant prompt files.
  2. Run this script with --grid-mode {none,grid,both}.
  3. Inspect the per-GT match report and the raw model output.
  4. Tune and rerun.

Nothing in src/ is modified. All state lives under:
  prompts/anchor_identification_v3_{nogrid,grid}_{system,user}.txt  (the prompts being tuned)
  data/areas/rookery_bay_v2/images/...                              (inputs, pre-fetched)
  data/areas/rookery_bay_v2/images/structures/root-10-8/...         (bundle + overlays)
  data/areas/rookery_bay_v2/images/z0_10_8_grid.png                 (grid overlay cache)
  data/areas/rookery_bay_v2/images/tuning_runs/                     (run logs, gitignored)

Usage:
  python scripts/tune_anchor_identification_v3.py                       # defaults to --grid-mode none
  python scripts/tune_anchor_identification_v3.py --grid-mode grid
  python scripts/tune_anchor_identification_v3.py --grid-mode both      # runs both, side-by-side report
  python scripts/tune_anchor_identification_v3.py --grid-mode none --replay
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_env = REPO_ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if not os.environ.get(_k):
            os.environ[_k] = _v

sys.path.insert(0, str(REPO_ROOT / "src"))

# ------------------------------------------------------------------
# Imports (after path fixup).
# ------------------------------------------------------------------

from readwater.api.claude_vision import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    _extract_json_from_response,
    _get_client,
    _load_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_anchor_id")

# ------------------------------------------------------------------
# Per-cell paths
# ------------------------------------------------------------------

IMG_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
RUNS_DIR = IMG_DIR / "tuning_runs"

# Default cell preserved so existing automation (and the original harness's
# command line) still works without --cell.
DEFAULT_CELL_ID = "root-10-8"


@dataclass
class CellPaths:
    """All input/output paths for a single tuning cell."""

    cell_id: str
    z16_image: Path
    z16_grid_overlay: Path
    struct_dir: Path
    bundle_json: Path
    overlay_z15: Path
    overlay_z14: Path
    overlay_z12: Path
    gt_anchors_json: Path

    @classmethod
    def for_cell(cls, cell_id: str) -> "CellPaths":
        # cell_id like "root-10-8" -> stem "z0_10_8"
        parts = cell_id.removeprefix("root-").split("-")
        if len(parts) != 2:
            raise ValueError(
                f"cell_id {cell_id!r} not recognized; expected 'root-<parent>-<child>'"
            )
        stem = "z0_" + "_".join(parts)
        struct_dir = IMG_DIR / "structures" / cell_id
        return cls(
            cell_id=cell_id,
            z16_image=IMG_DIR / f"{stem}.png",
            # NB: z0_X_Y_grid.png is the LEGACY 4x4 numbered grid produced by
            # the discover pipeline. Phase C uses a distinct 8x8 A1-H8 overlay
            # at z0_X_Y_grid_8x8.png so the two never collide.
            z16_grid_overlay=IMG_DIR / f"{stem}_grid_8x8.png",
            struct_dir=struct_dir,
            bundle_json=struct_dir / "context_bundle.json",
            overlay_z15=struct_dir / "overlay_z15_same_center.png",
            overlay_z14=struct_dir / "overlay_z14_parent.png",
            overlay_z12=struct_dir / "overlay_z12_grandparent.png",
            gt_anchors_json=struct_dir / "gt_anchors.json",
        )


# Module-level globals that exist purely so build_z16_bundles, the coord-gen
# harness, and quick interactive imports keep working unchanged. Always use
# CellPaths.for_cell() in new code.
_DEFAULT = CellPaths.for_cell(DEFAULT_CELL_ID)
CELL_ID = _DEFAULT.cell_id
Z16_IMAGE = _DEFAULT.z16_image
Z16_GRID_OVERLAY = _DEFAULT.z16_grid_overlay
STRUCT_DIR = _DEFAULT.struct_dir
BUNDLE_JSON = _DEFAULT.bundle_json
OVERLAY_Z15 = _DEFAULT.overlay_z15
OVERLAY_Z14 = _DEFAULT.overlay_z14
OVERLAY_Z12 = _DEFAULT.overlay_z12

# Grid mode constants.
#  - CLI values: "none" | "grid" | "both"  (per docs/PHASE_C_TASKS.md TASK-1)
#  - File-naming form: "nogrid" | "grid"   (matches prompts/anchor_identification_v3_<form>_*)
# _to_file_mode() translates between them.
GRID_MODE_NONE = "none"
GRID_MODE_GRID = "grid"
GRID_MODE_BOTH = "both"
CLI_MODES = (GRID_MODE_NONE, GRID_MODE_GRID, GRID_MODE_BOTH)
FILE_MODES = ("nogrid", "grid")


def _to_file_mode(cli_mode: str) -> str:
    if cli_mode == GRID_MODE_NONE:
        return "nogrid"
    if cli_mode == GRID_MODE_GRID:
        return "grid"
    raise ValueError(f"_to_file_mode does not handle {cli_mode!r}")


def _ensure_grid_overlay(paths: CellPaths | None = None) -> Path:
    """Render the 8x8 A1-H8 grid overlay on the cell's z16 image, cached on disk.

    Cache validation goes by IMAGE DIMENSIONS, not just mtime: a previous
    iteration of this codebase shipped a 4x4-numbered grid file at the same
    legacy path (`z0_X_Y_grid.png`), and a pure mtime check let those slip
    through as if they were 8x8 A1-H8 overlays. We now write to a distinct
    `_grid_8x8.png` filename AND reject any cache whose dimensions don't match
    the source image.
    """
    from PIL import Image as _PILImage

    from readwater.pipeline.structure.grid_overlay import (  # noqa: WPS433
        draw_label_grid,
        grid_shape_for_image,
    )

    p = paths or _DEFAULT
    if not p.z16_image.exists():
        raise FileNotFoundError(f"missing z16 source image: {p.z16_image}")

    with _PILImage.open(p.z16_image) as src_im:
        src_size = src_im.size

    cache_ok = False
    if p.z16_grid_overlay.exists() and p.z16_grid_overlay.stat().st_size > 0:
        try:
            with _PILImage.open(p.z16_grid_overlay) as cached:
                cache_ok = (cached.size == src_size)
        except Exception:  # noqa: BLE001 — corrupt cache = regenerate
            cache_ok = False
    if cache_ok:
        return p.z16_grid_overlay

    rows, cols = grid_shape_for_image(src_size, short_axis_cells=8)
    if (rows, cols) != (8, 8):
        # Defensive: a non-square z16 image would produce a non-(8,8) grid;
        # the prompt is hard-coded for A1-H8, so refuse to silently mismatch.
        raise ValueError(
            f"expected (8,8) grid for {p.cell_id} but got ({rows},{cols}); "
            f"source image dims = {src_size}"
        )
    draw_label_grid(str(p.z16_image), rows, cols, str(p.z16_grid_overlay))
    return p.z16_grid_overlay


# ------------------------------------------------------------------
# Ground truth for root-10-8
# ------------------------------------------------------------------

@dataclass
class GTAnchor:
    gt_id: str
    label: str
    types: set[str]                  # acceptable structure_type values
    tier: int | None                 # expected tier (None = any)
    location_keywords: list[set[str]]  # one set per "must include one of"; AND across sets
    required: bool = True
    should_be_hedged: bool = False   # needs_deeper_zoom true + confidence <= 0.5
    notes: str = ""


GROUND_TRUTH: list[GTAnchor] = [
    GTAnchor(
        gt_id="gt1",
        label="NW drain system",
        types={"drain_system", "creek_mouth_system"},
        tier=1,
        location_keywords=[{"north", "nw", "upper", "top"}],
        notes="Narrow cut in the N mangrove wall; water continues behind into interior",
    ),
    GTAnchor(
        gt_id="gt2",
        label="W lagoon upper island",
        types={"island"},
        tier=1,
        location_keywords=[{"west", "nw", "lagoon"}],
        notes="Small oval mangrove islet in the W back-barrier lagoon",
    ),
    GTAnchor(
        gt_id="gt3",
        label="W lagoon lower island",
        types={"island"},
        tier=1,
        location_keywords=[{"sw", "south", "lower"}, {"west", "lagoon", "channel"}],
        notes="Elongated N-S island in narrower S part of W lagoon",
    ),
    GTAnchor(
        gt_id="gt4",
        label="SE hammock island",
        types={"island"},
        tier=1,
        location_keywords=[{"se", "southeast", "east", "eastern"}],
        notes="Rounded triangular mangrove island with red interior hammock",
    ),
    GTAnchor(
        gt_id="gt5",
        label="peninsula N tip",
        types={"point"},
        tier=3,
        location_keywords=[{"peninsula", "tip", "center", "central", "middle"}],
        notes="N tip of the large central-south peninsula; peninsula itself is a zone",
    ),
    GTAnchor(
        gt_id="gt6",
        label="E shore trough",
        types={"trough", "current_split"},
        tier=3,
        location_keywords=[{"east", "eastern", "e shore", "e-shore"}],
        notes="Linear darker band along E mangrove shore — depth-break seam",
    ),
    GTAnchor(
        gt_id="gt7",
        label="ambiguous central lobe",
        types={"oyster_bar", "current_split", "trough", "island_edge", "shoreline_cut"},
        tier=None,
        location_keywords=[{"center", "central", "junction", "middle", "between"}],
        required=False,
        should_be_hedged=True,
        notes="Lighter-toned lobe in open basin — bottom type ambiguous",
    ),
    GTAnchor(
        gt_id="gt8",
        label="central junction island",
        types={"island"},
        tier=1,
        location_keywords=[{"center", "central", "junction", "middle", "basin"}],
        notes="Small (~50 m) rounded islet at 3-channel convergence",
    ),
]


# ------------------------------------------------------------------
# LLM call — 4-image + bundle
# ------------------------------------------------------------------


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_block(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _b64_png(path),
        },
    }


async def call_anchor_identification(
    file_mode: str,
    paths: CellPaths | None = None,
    inject_evidence: bool = False,
) -> tuple[dict, str]:
    """Run the v3 prompts against `paths.cell_id`. Returns (parsed_json, raw_text).

    file_mode is "nogrid" or "grid" — selects which prompt-pair to load and
    which version of image 1 (plain z16 vs grid-overlaid z16) to send.

    inject_evidence=True calls evidence.build_cell_evidence_section() to
    populate the {evidence_table} placeholder in the user prompt with the
    cell's NAIP water + NOAA channel + FWC oyster/seagrass coverage.
    Default is False so existing automation gets the same placeholder text
    (no evidence) without triggering a regression.
    """
    if file_mode not in FILE_MODES:
        raise ValueError(f"file_mode must be one of {FILE_MODES}, got {file_mode!r}")
    p = paths or _DEFAULT
    system_prompt = _load_prompt(f"anchor_identification_v3_{file_mode}_system.txt")
    user_template = _load_prompt(f"anchor_identification_v3_{file_mode}_user.txt")

    bundle_json = p.bundle_json.read_text(encoding="utf-8")
    # Load meta from bundle for the user template header.
    bundle = json.loads(bundle_json)
    self_lin = bundle["lineage"][-1]

    if inject_evidence:
        from readwater.pipeline.evidence import build_cell_evidence_section
        area_root = REPO_ROOT / "data" / "areas" / "rookery_bay_v2"
        evidence_table = build_cell_evidence_section(
            p.cell_id, area_root,
            grid_rows=8 if file_mode == "grid" else 8,
            grid_cols=8 if file_mode == "grid" else 8,
        )
    else:
        evidence_table = "(habitat evidence injection disabled for this run)"

    user_prompt = user_template.format(
        cell_id=p.cell_id,
        zoom=self_lin["zoom"],
        center_lat=f"{self_lin['center'][0]:.4f}",
        center_lon=f"{self_lin['center'][1]:.4f}",
        coverage_miles="0.37",
        context_bundle_json=bundle_json,
        evidence_table=evidence_table,
    )

    # Image 1 swaps in the grid-overlaid PNG when grid mode is on. Images 2-4
    # are unchanged.
    if file_mode == "grid":
        z16_path = _ensure_grid_overlay(p)
        image1_label = "IMAGE 1 — z16_local with 8x8 grid overlay (A1-H8):"
    else:
        z16_path = p.z16_image
        image1_label = "IMAGE 1 — z16_local (target cell, no overlay):"

    # Order matches the system prompt: 1=z16_local, 2=z15_same_center,
    # 3=z14_parent, 4=z12_grandparent.
    content = [
        {"type": "text", "text": image1_label},
        _image_block(z16_path),
        {"type": "text", "text": "IMAGE 2 — z15_same_center (yellow = z16 footprint):"},
        _image_block(p.overlay_z15),
        {"type": "text", "text": "IMAGE 3 — z14_parent (yellow = z16 footprint inside z14):"},
        _image_block(p.overlay_z14),
        {"type": "text", "text": "IMAGE 4 — z12_grandparent (yellow = z14 footprint inside z12):"},
        _image_block(p.overlay_z12),
        {"type": "text", "text": user_prompt},
    ]

    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw_text = response.content[0].text
    try:
        parsed = _extract_json_from_response(raw_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON extraction failed: %s", exc)
        parsed = {}
    return parsed, raw_text


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def _haystack_lower(text: str) -> str:
    """Lowercased haystack for substring matches; handles east→eastern etc."""
    return text.lower()


def _anchor_haystack(a: dict) -> str:
    """Everything readable on an anchor that could contain spatial words."""
    parts = [
        a.get("position_in_zone", ""),
        a.get("rationale", ""),
        a.get("flow_driver_or_structure", ""),
        " ".join(a.get("self_observations_cited", []) or []),
        " ".join(s.get("notes", "") for s in (a.get("sub_features") or [])),
        " ".join(s.get("name", "") for s in (a.get("sub_features") or [])),
    ]
    return " ".join(parts)


def _match_score(gt: GTAnchor, anchor: dict) -> float:
    """0.0 if this anchor clearly doesn't match, else a rough score."""
    stype = (anchor.get("structure_type") or "").strip()
    if stype not in gt.types:
        return 0.0
    # Tier check (soft — if GT has a required tier, prefer matches with that tier)
    tier_bonus = 0.0
    if gt.tier is not None:
        anchor_tier = anchor.get("tier")
        if anchor_tier == gt.tier:
            tier_bonus = 0.2
    # Location keyword check — AND across sets, OR within a set.
    # Substring match so "east" hits "eastern", "west" hits "western", etc.
    haystack = _haystack_lower(_anchor_haystack(anchor))
    for required_set in gt.location_keywords:
        if not any(kw in haystack for kw in required_set):
            return 0.0
    # Hedge check (only for should_be_hedged)
    hedge_bonus = 0.0
    if gt.should_be_hedged:
        if anchor.get("needs_deeper_zoom") and float(anchor.get("confidence") or 1.0) <= 0.6:
            hedge_bonus = 0.3
        else:
            return 0.0
    # Base match score = 1.0 + bonuses
    return 1.0 + tier_bonus + hedge_bonus


def match_ground_truth(model_anchors: list[dict]) -> list[dict]:
    """Greedy match: for each GT in order, pick the best unmatched anchor.

    Returns a list of result rows with gt_id, matched (bool), match_detail.
    """
    remaining = list(enumerate(model_anchors))
    results: list[dict] = []
    for gt in GROUND_TRUTH:
        best_score = 0.0
        best_idx = -1
        for ridx, anchor in remaining:
            score = _match_score(gt, anchor)
            if score > best_score:
                best_score = score
                best_idx = ridx
        if best_idx >= 0:
            # Pull the matched anchor out so it can't be double-matched.
            remaining = [(i, a) for (i, a) in remaining if i != best_idx]
            matched_anchor = model_anchors[best_idx]
            results.append({
                "gt_id": gt.gt_id,
                "label": gt.label,
                "required": gt.required,
                "matched": True,
                "score": best_score,
                "anchor_id": matched_anchor.get("anchor_id"),
                "matched_type": matched_anchor.get("structure_type"),
                "matched_tier": matched_anchor.get("tier"),
                "matched_position": matched_anchor.get("position_in_zone"),
            })
        else:
            results.append({
                "gt_id": gt.gt_id,
                "label": gt.label,
                "required": gt.required,
                "matched": False,
                "score": 0.0,
            })
    # Also report unmatched model anchors (false positives).
    unmatched_model = [model_anchors[i] for (i, _) in remaining]
    return results, unmatched_model


# ------------------------------------------------------------------
# Run + report
# ------------------------------------------------------------------


def _emoji(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_report(
    parsed: dict, gt_results: list[dict], unmatched: list[dict],
    cell_id: str | None = None,
) -> None:
    print()
    print("=" * 76)
    print(f"ANCHOR IDENTIFICATION TUNING REPORT — {cell_id or CELL_ID}")
    print("=" * 76)
    zones = parsed.get("zones") or []
    print(f"zones: {len(zones)}")
    for z in zones:
        print(f"  - {z.get('zone_id')}: {z.get('label')}  ({z.get('footprint_in_cell')})")
    sweep = parsed.get("sweep_notes") or {}
    if sweep:
        print("sweep_notes:", sweep)
    anchors = parsed.get("anchors") or []
    print()
    print(f"model returned {len(anchors)} anchors:")
    for a in anchors:
        print(
            f"  - {a.get('anchor_id')}  T{a.get('tier')}  {a.get('structure_type'):<22}"
            f"  size={a.get('size_fraction')}  pos={(a.get('position_in_zone') or '')[:60]}"
        )
    print()
    print("ground-truth match report:")
    required_total = sum(1 for r in gt_results if r["required"])
    required_hit = sum(1 for r in gt_results if r["required"] and r["matched"])
    optional_hit = sum(1 for r in gt_results if (not r["required"]) and r["matched"])
    for r in gt_results:
        req_tag = "REQ" if r["required"] else "opt"
        status = _emoji(r["matched"])
        if r["matched"]:
            detail = (
                f" -> {r['anchor_id']} {r['matched_type']} T{r['matched_tier']} "
                f"\"{(r['matched_position'] or '')[:50]}\""
            )
        else:
            detail = " -> (no match)"
        print(f"  [{req_tag}] {status}  {r['gt_id']}  {r['label']}{detail}")
    print()
    print(f"REQUIRED: {required_hit}/{required_total} "
          f"({required_hit / required_total * 100:.0f}%)  "
          f"optional: {optional_hit}/1")
    if unmatched:
        print()
        print(f"unmatched model anchors (possible false positives): {len(unmatched)}")
        for a in unmatched:
            print(
                f"  - {a.get('anchor_id')}  T{a.get('tier')}  {a.get('structure_type')}  "
                f"pos={(a.get('position_in_zone') or '')[:60]}"
            )
    print("=" * 76)


def save_run(
    file_mode: str,
    parsed: dict,
    raw_text: str,
    gt_results: list[dict],
    unmatched: list[dict],
    paths: CellPaths | None = None,
) -> Path:
    p = paths or _DEFAULT
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"v3_{p.cell_id}_{file_mode}_{ts}"
    out.mkdir()
    (out / "raw_response.md").write_text(raw_text, encoding="utf-8")
    (out / "parsed.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    (out / "gt_report.json").write_text(json.dumps({
        "version": 3,
        "grid_mode": file_mode,
        "cell_id": p.cell_id,
        "gt_results": gt_results,
        "unmatched_model": unmatched,
    }, indent=2), encoding="utf-8")
    return out


def _find_latest_run(file_mode: str, paths: CellPaths | None = None) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    p = paths or _DEFAULT
    # Try cell-tagged dirs first (new naming).
    candidates = sorted(RUNS_DIR.glob(f"v3_{p.cell_id}_{file_mode}_*"), reverse=True)
    if candidates:
        return candidates[0]
    # Fallback to legacy untagged naming for root-10-8.
    if p.cell_id == DEFAULT_CELL_ID:
        legacy = sorted(RUNS_DIR.glob(f"v3_{file_mode}_*"), reverse=True)
        # Filter out the cell-tagged ones (which would also match the legacy glob).
        legacy = [c for c in legacy if not c.name.startswith(f"v3_{p.cell_id}_")]
        if legacy:
            return legacy[0]
    return None


def print_side_by_side(
    nogrid: tuple[dict, list[dict], list[dict]],
    grid: tuple[dict, list[dict], list[dict]],
) -> None:
    """Compact head-to-head report for --grid-mode both."""
    np_, ng_results, ng_unmatched = nogrid
    gp, g_results, g_unmatched = grid

    def _hits(results: list[dict]) -> tuple[int, int, int]:
        req_total = sum(1 for r in results if r["required"])
        req_hit = sum(1 for r in results if r["required"] and r["matched"])
        opt_hit = sum(1 for r in results if (not r["required"]) and r["matched"])
        return req_total, req_hit, opt_hit

    ng_req_total, ng_req, ng_opt = _hits(ng_results)
    g_req_total, g_req, g_opt = _hits(g_results)
    ng_anchors = len((np_.get("anchors") or []))
    g_anchors = len((gp.get("anchors") or []))

    print()
    print("=" * 76)
    print(f"GRID-MODE SIDE-BY-SIDE — {CELL_ID}")
    print("=" * 76)
    print(f"{'metric':<28s}  {'nogrid':>10s}  {'grid':>10s}")
    print("-" * 56)
    print(f"{'anchors emitted':<28s}  {ng_anchors:>10d}  {g_anchors:>10d}")
    print(f"{'required GT hits':<28s}  {ng_req:>5d}/{ng_req_total:<4d}  {g_req:>5d}/{g_req_total:<4d}")
    print(f"{'optional GT hits':<28s}  {ng_opt:>10d}  {g_opt:>10d}")
    print(f"{'unmatched (false pos)':<28s}  {len(ng_unmatched):>10d}  {len(g_unmatched):>10d}")
    print()
    print("per-GT agreement:")
    by_gt_ng = {r["gt_id"]: r for r in ng_results}
    by_gt_g = {r["gt_id"]: r for r in g_results}
    all_ids = list(by_gt_ng.keys())
    print(f"  {'gt_id':<6s}  {'label':<40s}  {'nogrid':<8s}  {'grid':<8s}")
    for gt_id in all_ids:
        rn = by_gt_ng[gt_id]
        rg = by_gt_g.get(gt_id, {"matched": False, "label": rn["label"]})
        ng_tag = "HIT" if rn["matched"] else "MISS"
        g_tag = "HIT" if rg["matched"] else "MISS"
        agree = " " if ng_tag == g_tag else " <-- diverge"
        print(f"  {gt_id:<6s}  {rn['label'][:40]:<40s}  {ng_tag:<8s}  {g_tag:<8s}{agree}")
    print("=" * 76)


async def _run_one(
    file_mode: str, replay: bool, paths: CellPaths | None = None,
    inject_evidence: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    """Execute (or replay) a single file_mode and return (parsed, gt_results, unmatched).

    file_mode is "nogrid" or "grid" — caller is responsible for translating
    from CLI-form ("none"/"grid") via _to_file_mode().
    """
    if file_mode not in FILE_MODES:
        raise ValueError(f"file_mode must be one of {FILE_MODES}, got {file_mode!r}")
    p = paths or _DEFAULT
    if replay:
        run_dir = _find_latest_run(file_mode, p)
        if run_dir is None:
            print(f"no v3_{p.cell_id}_{file_mode} runs on disk to replay")
            sys.exit(2)
        parsed = json.loads((run_dir / "parsed.json").read_text(encoding="utf-8"))
        raw_text = (run_dir / "raw_response.md").read_text(encoding="utf-8")
        logger.info("replaying %s", run_dir.name)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set; aborting.")
            sys.exit(2)
        for required in (
            p.z16_image, p.overlay_z15, p.overlay_z14, p.overlay_z12, p.bundle_json,
        ):
            if not required.exists():
                print(f"missing input for {p.cell_id}: {required}")
                sys.exit(2)
        logger.info("calling claude with v3 %s prompts (cell %s, evidence=%s)...",
                    file_mode, p.cell_id, inject_evidence)
        parsed, raw_text = await call_anchor_identification(
            file_mode, p, inject_evidence=inject_evidence,
        )

    anchors = parsed.get("anchors") if isinstance(parsed, dict) else None
    if not isinstance(anchors, list):
        anchors = []

    # Python GROUND_TRUTH only covers root-10-8. For other cells, skip
    # keyword-based scoring; the JSON-GT-based tier-weighted scoring runs
    # in the separate sweep / coverage script.
    if p.cell_id == DEFAULT_CELL_ID:
        gt_results, unmatched = match_ground_truth(anchors)
    else:
        gt_results = []
        unmatched = list(anchors)
    print_report(parsed, gt_results, unmatched, cell_id=p.cell_id)
    if not replay:
        out_dir = save_run(file_mode, parsed, raw_text, gt_results, unmatched, p)
        print(f"\nartifacts saved to: {out_dir.relative_to(REPO_ROOT)}")
    return parsed, gt_results, unmatched


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell", type=str, default=DEFAULT_CELL_ID,
        help=f"Which cell to tune against (default {DEFAULT_CELL_ID})",
    )
    parser.add_argument(
        "--grid-mode",
        choices=CLI_MODES,
        default=GRID_MODE_NONE,
        help="Which v3 variant to run. 'both' runs nogrid + grid sequentially "
             "and emits a side-by-side report.",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Re-score the most recent run on disk without calling the API. "
             "With --grid-mode both, replays the latest of each.",
    )
    parser.add_argument(
        "--evidence", action="store_true",
        help="Inject NAIP/NOAA/FWC habitat evidence table into the v3 user "
             "prompt (TASK-3). Off by default.",
    )
    args = parser.parse_args()

    cell_paths = CellPaths.for_cell(args.cell)

    if args.grid_mode == GRID_MODE_BOTH:
        # Run nogrid then grid (sequential keeps API rate-limit pressure low).
        nogrid_result = await _run_one("nogrid", args.replay, cell_paths,
                                       inject_evidence=args.evidence)
        grid_result = await _run_one("grid", args.replay, cell_paths,
                                     inject_evidence=args.evidence)
        print_side_by_side(nogrid_result, grid_result)
    else:
        await _run_one(_to_file_mode(args.grid_mode), args.replay, cell_paths,
                       inject_evidence=args.evidence)


if __name__ == "__main__":
    asyncio.run(main())
