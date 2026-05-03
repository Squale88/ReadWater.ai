"""CV-feature orchestrator (Phase 1: dedup + combined render).

Reads the latest JSON output from each per-feature detector for a cell:
  cv_detect_drains.py   →  cv_drains_<ts>.json   (DRAIN / CREEK_MOUTH / LARGE_POCKET / SHOAL)
  cv_detect_islands.py  →  cv_islands_<ts>.json  (ISLAND_SMALL/MEDIUM/LARGE)
  cv_detect_points.py   →  cv_points_<ts>.json   (POINT_R13 / POINT_R26)
  cv_detect_pockets.py  →  cv_pockets_<ts>.json  (POCKET_R13 / POCKET_R26)

Drops "true duplicates" (same physical feature detected by two detectors at the
same place, or detected at two scales of one detector), then renders a single
combined overlay PNG showing every kept candidate color-coded by category.
Writes a combined JSON listing every kept candidate.

Phase 1 deliberately stops here — no anchor hierarchy (primary/secondary/
tertiary), no habitat overlay. Those land in Phase 2 and Phase 3.

DEDUP MODEL

Two candidates are "duplicates" iff:
  (1) their pixel_centers are within DEDUP_DISTANCE_PX of each other, AND
  (2) they belong to the same compatibility group.

Compatibility groups (features that COULD describe the same physical thing):
  Group W = narrow-water features
            (DRAIN, CREEK_MOUTH, LARGE_POCKET, SHOAL, POCKET_R13, POCKET_R26)
  Group L = narrow-land features (POINT_R13, POINT_R26)
  Group I = land bodies (ISLAND_SMALL, ISLAND_MEDIUM, ISLAND_LARGE)

Cross-group pairs are NOT dedup candidates (water features and land features
can't be the same physical thing even if they're at the same pixel center).

Within a duplicate pair, the higher-priority category wins. CATEGORY_PRIORITY
defines the order. Priority is roughly "more comprehensive / more informative"
first; e.g. DRAIN > LARGE_POCKET (a drain says more than just "pocket exists")
and POINT_R26 > POINT_R13 (R26 catches the same shape with more pixels).

Input:  data/areas/rookery_bay_v2/images/structures/<cell>/cv_drains_*.json
        data/areas/rookery_bay_v2/images/structures/<cell>/cv_islands_*.json
        data/areas/rookery_bay_v2/images/structures/<cell>/cv_points_*.json
        data/areas/rookery_bay_v2/images/structures/<cell>/cv_pockets_*.json
Output: data/areas/rookery_bay_v2/images/structures/<cell>/cv_all_<ts>.{png,json}

Usage:
  python scripts/cv_detect_all.py --cell root-10-8
  python scripts/cv_detect_all.py --cell root-10-8 --cell root-11-5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cv_helpers import (  # noqa: E402
    REPO_ROOT,
    grid_cell_for,
    load_font,
)

# ---- Dedup parameters ----

DEDUP_DISTANCE_PX = 15        # centers within this many px = same physical feature

# Categories grouped by what they describe physically. Only intra-group pairs
# can be duplicates.
GROUP_WATER_NARROW = {
    "DRAIN", "CREEK_MOUTH", "LARGE_POCKET", "SHOAL",
    "POCKET_R13", "POCKET_R26",
}
GROUP_LAND_NARROW = {"POINT_R13", "POINT_R26"}
GROUP_LAND_BODY = {"ISLAND_SMALL", "ISLAND_MEDIUM", "ISLAND_LARGE"}


def category_group(category: str) -> str | None:
    if category in GROUP_WATER_NARROW: return "W"
    if category in GROUP_LAND_NARROW:  return "L"
    if category in GROUP_LAND_BODY:    return "I"
    return None


# Priority for dedup tie-breaking. Higher = preferred when categories collide
# at the same physical location. Roughly "more informative" first.
CATEGORY_PRIORITY = {
    # narrow water (drains beats pockets when they describe the same throat)
    "DRAIN":         100,
    "CREEK_MOUTH":    90,
    "LARGE_POCKET":   80,
    "SHOAL":          70,
    "POCKET_R26":     60,
    "POCKET_R13":     50,
    # land bodies (larger tier wins; in practice same island can't be in
    # two tiers, but list anyway for completeness)
    "ISLAND_LARGE":   85,
    "ISLAND_MEDIUM":  75,
    "ISLAND_SMALL":   55,
    # narrow land (R26 catches same shape with more pixels than R13)
    "POINT_R26":      45,
    "POINT_R13":      35,
}


# ---- Per-category visual rendering (consistent across detectors) ----

CATEGORY_COLOR = {
    "DRAIN":         (220,  30,  30, 255),    # red
    "CREEK_MOUTH":   (255, 165,   0, 255),    # orange
    "LARGE_POCKET":  (255, 230,  50, 255),    # yellow
    "SHOAL":         ( 60, 200, 230, 255),    # cyan-ish
    "ISLAND_SMALL":  (150, 230, 100, 255),    # light green
    "ISLAND_MEDIUM": ( 50, 180,  50, 255),    # medium green
    "ISLAND_LARGE":  ( 30, 110,  30, 255),    # dark green
    "POINT_R13":     (255, 150, 200, 255),    # pink
    "POINT_R26":     (220,  80, 180, 255),    # magenta
    "POCKET_R13":    (140, 200, 255, 255),    # light blue
    "POCKET_R26":    ( 30, 110, 200, 255),    # deep blue
}


# ---- IO ----


STRUCT_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images" / "structures"
SAT_DIR    = REPO_ROOT / "data" / "areas" / "rookery_bay_v2" / "images"


def latest_json(cell_id: str, prefix: str) -> Path | None:
    """Return the newest <prefix>*.json file in the cell's structures dir, or None."""
    d = STRUCT_DIR / cell_id
    if not d.exists():
        return None
    matches = sorted(d.glob(f"{prefix}*.json"))
    return matches[-1] if matches else None


def load_detector(cell_id: str, prefix: str, source: str) -> list[dict]:
    """Load the latest detector JSON and normalize each candidate to a common
    record. Returns [] if the detector hasn't been run for this cell.
    """
    p = latest_json(cell_id, prefix)
    if p is None:
        print(f"  [warn] {cell_id}: no {prefix}*.json found, skipping {source}")
        return []
    raw = json.loads(p.read_text())
    out: list[dict] = []
    for c in raw.get("candidates", []):
        out.append({
            "id": c["id"],                          # detector-local id (c1, i1, p1, k1)
            "category": c["category"],
            "source_detector": source,
            "source_path": str(p),
            "pixel_bbox": list(c["pixel_bbox"]),
            "pixel_center": list(c["pixel_center"]),
            "extra": c,                             # preserve everything for downstream
        })
    return out


# ---- Dedup ----


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _winner(a: dict, b: dict) -> tuple[dict, dict]:
    """Return (kept, dropped) for a duplicate pair. Higher CATEGORY_PRIORITY
    wins; tie-breaker is larger bbox area."""
    pa = CATEGORY_PRIORITY.get(a["category"], 0)
    pb = CATEGORY_PRIORITY.get(b["category"], 0)
    if pa > pb: return a, b
    if pb > pa: return b, a
    # priority tie — prefer the one with the larger bbox
    if _bbox_area(a["pixel_bbox"]) >= _bbox_area(b["pixel_bbox"]):
        return a, b
    return b, a


def dedup_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (kept, dropped). A candidate is dropped if a duplicate of higher
    priority is found within DEDUP_DISTANCE_PX in the same compatibility group.

    O(n²) sweep — fine for the per-cell scale (≤ ~150 candidates).
    """
    n = len(candidates)
    drop_idx: set[int] = set()
    drop_reasons: list[tuple[int, dict]] = []   # (index, paired_winner_id)

    for i in range(n):
        if i in drop_idx:
            continue
        ci = candidates[i]
        gi = category_group(ci["category"])
        if gi is None:
            continue
        cxi, cyi = ci["pixel_center"]
        for j in range(i + 1, n):
            if j in drop_idx:
                continue
            cj = candidates[j]
            if category_group(cj["category"]) != gi:
                continue
            cxj, cyj = cj["pixel_center"]
            if (cxi - cxj) ** 2 + (cyi - cyj) ** 2 > DEDUP_DISTANCE_PX ** 2:
                continue
            kept, dropped = _winner(ci, cj)
            kept_id = id(kept)
            dropped_idx = j if id(cj) == id(dropped) else i
            drop_idx.add(dropped_idx)
            drop_reasons.append((dropped_idx, kept))
            if dropped_idx == i:
                # 'i' itself got dropped; stop scanning further pairs against it
                break

    kept_list = [c for k, c in enumerate(candidates) if k not in drop_idx]
    dropped_list = []
    for k_idx, winner in drop_reasons:
        loser = candidates[k_idx].copy()
        loser["dropped_for"] = {
            "winner_source": winner["source_detector"],
            "winner_id": winner["id"],
            "winner_category": winner["category"],
        }
        dropped_list.append(loser)
    return kept_list, dropped_list


# ---- Rendering ----


LEGEND_PANEL_HEIGHT = 320     # px below the satellite for the off-image legend
LEGEND_FONT_SIZE = 13


def render_combined_overlay(base_image_path: Path,
                            kept: list[dict],
                            dropped: list[dict],
                            output_path: Path,
                            image_size: tuple[int, int] = (1280, 1280)) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    canvas = Image.new("RGBA", (image_size[0], image_size[1] + LEGEND_PANEL_HEIGHT),
                       (20, 20, 20, 255))
    canvas.paste(base, (0, 0))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    # 8x8 A1-H8 grid (over the satellite portion only)
    rows, cols = 8, 8
    gw, gh = image_size
    cell_w, cell_h = gw / cols, gh / rows
    for i in range(1, cols):
        x = int(i * cell_w)
        draw.line([(x + 1, 0), (x + 1, gh)], fill=(0, 0, 0, 200), width=1)
        draw.line([(x, 0), (x, gh)], fill=(255, 255, 255, 230), width=2)
    for j in range(1, rows):
        y = int(j * cell_h)
        draw.line([(0, y + 1), (gw, y + 1)], fill=(0, 0, 0, 200), width=1)
        draw.line([(0, y), (gw, y)], fill=(255, 255, 255, 230), width=2)
    grid_font = load_font(max(10, int(min(cell_w, cell_h) * 0.30)))
    for r in range(rows):
        for c in range(cols):
            text = f"{chr(ord('A') + r)}{c + 1}"
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            bbox_t = grid_font.getbbox(text)
            tw = bbox_t[2] - bbox_t[0]
            th = bbox_t[3] - bbox_t[1]
            tx = cx - tw // 2
            ty = cy - th // 2
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), text, fill="black", font=grid_font)
            draw.text((tx, ty), text, fill="white", font=grid_font)

    # Plot each kept candidate as a colored dot + short id label
    label_font = load_font(13)
    for cand in kept:
        cx, cy = cand["pixel_center"]
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=color, outline=(255, 255, 255, 255), width=2)
        # Label: just the source-detector id (already short — c1, i3, p7, k2)
        text = cand["id"]
        tx = int(cx) + 8
        ty = int(cy) - 14
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
        draw.text((tx, ty), text, fill=color, font=label_font)

    Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB").paste(
        Image.new("RGB", base.size), (0, 0))   # noop, just to keep base shape
    composed_top = Image.alpha_composite(base, layer)
    canvas.paste(composed_top, (0, 0))

    # ---- Off-satellite legend panel ----
    legend_draw = ImageDraw.Draw(canvas)
    panel_y0 = image_size[1]
    legend_font = load_font(LEGEND_FONT_SIZE)
    title_font = load_font(15)

    # Header row
    legend_draw.text((10, panel_y0 + 8),
                     f"Phase 1: dedup-only.  Kept = {len(kept)}, dropped = {len(dropped)} "
                     f"(dedup distance {DEDUP_DISTANCE_PX} px)",
                     fill=(255, 255, 255, 255), font=title_font)

    # Two columns: kept (left), dropped (right)
    col_w = image_size[0] // 2
    line_h = LEGEND_FONT_SIZE + 4
    header_y = panel_y0 + 32
    legend_draw.text((10, header_y), "KEPT", fill=(180, 255, 180, 255), font=title_font)
    legend_draw.text((col_w + 10, header_y), "DROPPED (duplicates)",
                     fill=(255, 180, 180, 255), font=title_font)

    # Sort kept by category priority (highest first), then by id
    kept_sorted = sorted(
        kept,
        key=lambda c: (-CATEGORY_PRIORITY.get(c["category"], 0), c["source_detector"], c["id"]),
    )
    y = header_y + line_h + 4
    max_lines = (LEGEND_PANEL_HEIGHT - (y - panel_y0) - 4) // line_h
    for cand in kept_sorted[:max_lines]:
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        cx, cy = cand["pixel_center"]
        cell = grid_cell_for(cx, cy, image_size)
        # Color swatch
        legend_draw.rectangle([10, y + 2, 22, y + LEGEND_FONT_SIZE],
                              fill=color)
        # Text
        line = (f"{cand['source_detector'][:1]}{cand['id']}  {cand['category']:<14s}  "
                f"@{cell}  ({int(cx)},{int(cy)})")
        legend_draw.text((28, y), line,
                         fill=(255, 255, 255, 255), font=legend_font)
        y += line_h
    if len(kept_sorted) > max_lines:
        legend_draw.text((28, y),
                         f"... +{len(kept_sorted) - max_lines} more (see JSON)",
                         fill=(180, 180, 180, 255), font=legend_font)

    # Dropped column
    y = header_y + line_h + 4
    for cand in dropped:
        if (y - panel_y0) > LEGEND_PANEL_HEIGHT - line_h - 4:
            legend_draw.text((col_w + 28, y),
                             f"... +{len(dropped) - ((y - header_y - line_h - 4) // line_h)} more",
                             fill=(180, 180, 180, 255), font=legend_font)
            break
        color = CATEGORY_COLOR.get(cand["category"], (200, 200, 200, 255))
        cx, cy = cand["pixel_center"]
        cell = grid_cell_for(cx, cy, image_size)
        legend_draw.rectangle([col_w + 10, y + 2, col_w + 22, y + LEGEND_FONT_SIZE],
                              fill=color)
        winner = cand.get("dropped_for", {})
        line = (f"{cand['id']} {cand['category']:<13s} @{cell} "
                f"-> {winner.get('winner_id', '?')}/{winner.get('winner_category', '?')}")
        legend_draw.text((col_w + 28, y), line,
                         fill=(255, 255, 255, 255), font=legend_font)
        y += line_h

    canvas.convert("RGB").save(output_path)
    return output_path


# ---- CLI ----


def run_one(cell_id: str) -> int:
    parent_num, child_num = cell_id.removeprefix("root-").split("-")
    stem = f"z0_{parent_num}_{child_num}"
    z16_path = SAT_DIR / f"{stem}.png"
    out_dir = STRUCT_DIR / cell_id

    if not z16_path.exists():
        print(f"{cell_id}: missing z16 satellite at {z16_path}, skipping.")
        return 1

    print(f"--- {cell_id} ---")

    # Load each detector's latest output
    by_source = [
        ("drains",  "cv_drains_"),
        ("islands", "cv_islands_"),
        ("points",  "cv_points_"),
        ("pockets", "cv_pockets_"),
    ]
    candidates: list[dict] = []
    per_source_counts: dict[str, int] = {}
    for source, prefix in by_source:
        loaded = load_detector(cell_id, prefix, source)
        per_source_counts[source] = len(loaded)
        candidates.extend(loaded)

    print(f"  loaded:  " + "  ".join(f"{s}={n}" for s, n in per_source_counts.items())
          + f"  -> {len(candidates)} total")

    kept, dropped = dedup_candidates(candidates)
    print(f"  dedup:   {len(kept)} kept,  {len(dropped)} dropped "
          f"(threshold {DEDUP_DISTANCE_PX} px)")

    if dropped:
        print(f"\n  Dropped duplicates (loser -> winner):")
        for d in dropped:
            w = d["dropped_for"]
            print(f"    {d['source_detector'][:7]:<7s} {d['id']:>4s} {d['category']:<14s}"
                  f"  ->  {w['winner_source'][:7]:<7s} {w['winner_id']:>4s} {w['winner_category']}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = out_dir / f"cv_all_{ts}.png"
    json_path = out_dir / f"cv_all_{ts}.json"

    render_combined_overlay(z16_path, kept, dropped, overlay_path)

    json_path.write_text(json.dumps({
        "cell_id": cell_id,
        "phase": 1,
        "dedup_distance_px": DEDUP_DISTANCE_PX,
        "loaded_per_source": per_source_counts,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "kept": [
            {k: v for k, v in c.items() if k != "extra"}
            | {"extra": c["extra"]}
            for c in kept
        ],
        "dropped": dropped,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\n  overlay: {overlay_path.relative_to(REPO_ROOT)}")
    print(f"  json:    {json_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", required=True,
                        help="Cell id like root-10-8. Repeat for multiple.")
    args = parser.parse_args()
    rc_overall = 0
    for cell_id in args.cell:
        rc = run_one(cell_id)
        if rc != 0:
            rc_overall = rc
    return rc_overall


if __name__ == "__main__":
    sys.exit(main())
