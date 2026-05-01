"""Iteration harness for tuning anchor_identification_v{N} prompts.

Drives the current prompts against a single tuning cell (root-10-8), calls
Claude with the 4-image handoff + context bundle, parses the JSON output,
and scores it against a hand-labeled ground truth for that cell.

Intended workflow:
  1. Edit prompts/anchor_identification_{system,user}_vN.txt.
  2. Run this script.
  3. Inspect the per-GT match report and the raw model output.
  4. Tune the prompts and rerun.

Nothing in src/ is modified. All state lives under:
  prompts/anchor_identification_{system,user}_v{VERSION}.txt       (the prompt being tuned)
  data/areas/rookery_bay_v2/images/...                             (inputs, pre-fetched)
  data/areas/rookery_bay_v2/images/structures/root-10-8/...        (bundle + overlays)
  data/areas/rookery_bay_v2/images/tuning_runs/                    (run logs, gitignored)

Usage:
  python scripts/tune_anchor_identification_v3.py
  python scripts/tune_anchor_identification_v3.py --version 3
  python scripts/tune_anchor_identification_v3.py --version 4  (after creating v4 files)
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
# Paths for the tuning cell
# ------------------------------------------------------------------

IMG_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"
CELL_ID = "root-10-8"
Z16_IMAGE = IMG_DIR / "z0_10_8.png"
STRUCT_DIR = IMG_DIR / "structures" / CELL_ID
BUNDLE_JSON = STRUCT_DIR / "context_bundle.json"
OVERLAY_Z15 = STRUCT_DIR / "overlay_z15_same_center.png"
OVERLAY_Z14 = STRUCT_DIR / "overlay_z14_parent.png"
OVERLAY_Z12 = STRUCT_DIR / "overlay_z12_grandparent.png"

RUNS_DIR = IMG_DIR / "tuning_runs"


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


async def call_anchor_identification(version: int) -> tuple[dict, str]:
    """Run the v{version} prompts against root-10-8. Returns (parsed_json, raw_text)."""
    system_prompt = _load_prompt(f"anchor_identification_system_v{version}.txt")
    user_template = _load_prompt(f"anchor_identification_user_v{version}.txt")

    bundle_json = BUNDLE_JSON.read_text(encoding="utf-8")
    # Load meta from bundle for the user template header.
    bundle = json.loads(bundle_json)
    self_lin = bundle["lineage"][-1]
    user_prompt = user_template.format(
        cell_id=CELL_ID,
        zoom=self_lin["zoom"],
        center_lat=f"{self_lin['center'][0]:.4f}",
        center_lon=f"{self_lin['center'][1]:.4f}",
        coverage_miles="0.37",
        context_bundle_json=bundle_json,
    )

    # Order matches the system prompt: 1=z16_local, 2=z15_same_center,
    # 3=z14_parent, 4=z12_grandparent.
    content = [
        {"type": "text", "text": "IMAGE 1 — z16_local (target cell, no overlay):"},
        _image_block(Z16_IMAGE),
        {"type": "text", "text": "IMAGE 2 — z15_same_center (yellow = z16 footprint):"},
        _image_block(OVERLAY_Z15),
        {"type": "text", "text": "IMAGE 3 — z14_parent (yellow = z16 footprint inside z14):"},
        _image_block(OVERLAY_Z14),
        {"type": "text", "text": "IMAGE 4 — z12_grandparent (yellow = z14 footprint inside z12):"},
        _image_block(OVERLAY_Z12),
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


def print_report(parsed: dict, gt_results: list[dict], unmatched: list[dict]) -> None:
    print()
    print("=" * 76)
    print(f"ANCHOR IDENTIFICATION TUNING REPORT — {CELL_ID}")
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


def save_run(version: int, parsed: dict, raw_text: str, gt_results: list[dict], unmatched: list[dict]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"v{version}_{ts}"
    out.mkdir()
    (out / "raw_response.md").write_text(raw_text, encoding="utf-8")
    (out / "parsed.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    (out / "gt_report.json").write_text(json.dumps({
        "version": version,
        "cell_id": CELL_ID,
        "gt_results": gt_results,
        "unmatched_model": unmatched,
    }, indent=2), encoding="utf-8")
    return out


def _find_latest_run(version: int) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob(f"v{version}_*"), reverse=True)
    return candidates[0] if candidates else None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument(
        "--replay", action="store_true",
        help="Re-score the most recent run on disk without calling the API",
    )
    args = parser.parse_args()

    if args.replay:
        run_dir = _find_latest_run(args.version)
        if run_dir is None:
            print(f"no v{args.version} runs on disk to replay")
            sys.exit(2)
        parsed = json.loads((run_dir / "parsed.json").read_text(encoding="utf-8"))
        raw_text = (run_dir / "raw_response.md").read_text(encoding="utf-8")
        logger.info("replaying %s", run_dir.name)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set; aborting.")
            sys.exit(2)
        for p in (Z16_IMAGE, OVERLAY_Z15, OVERLAY_Z14, OVERLAY_Z12, BUNDLE_JSON):
            if not p.exists():
                print(f"missing input: {p}")
                sys.exit(2)
        logger.info("calling claude with v%d prompts...", args.version)
        parsed, raw_text = await call_anchor_identification(args.version)

    anchors = parsed.get("anchors") if isinstance(parsed, dict) else None
    if not isinstance(anchors, list):
        anchors = []

    gt_results, unmatched = match_ground_truth(anchors)
    print_report(parsed, gt_results, unmatched)
    if not args.replay:
        out_dir = save_run(args.version, parsed, raw_text, gt_results, unmatched)
        print(f"\nartifacts saved to: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
