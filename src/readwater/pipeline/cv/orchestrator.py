"""CV-feature orchestrator (Phase 1: dedup + Phase 2: anchor clustering +
Phase 3a: habitat + Phase 3b: anchor-of-anchors parent/child links).

Reads the latest JSON output from each per-feature detector for a cell:
  cv_detect_drains.py   ->  cv_drains_<ts>.json   (DRAIN / CREEK_MOUTH / LARGE_POCKET / SHOAL)
  cv_detect_islands.py  ->  cv_islands_<ts>.json  (ISLAND_SMALL/MEDIUM/LARGE)
  cv_detect_points.py   ->  cv_points_<ts>.json   (POINT_R13 / POINT_R26)
  cv_detect_pockets.py  ->  cv_pockets_<ts>.json  (POCKET_R13 / POCKET_R26)

Plus, as of Phase 3a, runs habitat detection inline (no separate detector
script) on the FWC SAV / oyster-reef rasters loaded via
``readwater.storage`` from ``data/areas/<area>/masks/{seagrass,oyster}/``.

PIPELINE

  Phase 1 (dedup): drop "true duplicates" — same physical feature detected
  by two detectors at the same place, or detected at two scales of one
  detector.

  Phase 2 (cluster): group surviving candidates into ANCHORs. Each anchor
  has one PRIMARY (the largest feature in the group), zero or more
  SECONDARY (other features touching or near the primary's bbox), and
  zero or more TERTIARY references (feature ids that belong to OTHER
  anchors but happen to be within the tertiary radius of this anchor's
  primary center).

  Phase 3a (habitat): seagrass beds and oyster reefs feed the same dedup +
  clustering pipeline as a regular candidate stream, but with two design
  rules that protect against habitat dominating structural reality:

    (i) Seagrass primary priority is BELOW every structural category, so
        a structural feature (island / drain / creek / pocket / point /
        shoal) always claims primary status when present in a cluster.
        Seagrass only becomes primary when it has no structural neighbor
        in range — i.e., a true open seagrass flat with nothing else.
   (ii) An anchor's union bbox is capped at ANCHOR_FOOTPRINT_CAP_PX on
        either axis (~3x3 grid cells + 10%). When attaching a candidate
        as secondary would push the cluster past the cap, the candidate
        is skipped and remains free to become its own anchor.

  Anchors split into two flavors:
    - STRUCTURAL anchors (id "a1", "a2", ...): primary is a structural
      candidate (or an oyster, since oyster never anchors a meaningful
      cluster on its own).
    - HABITAT anchors (id "h1", "h2", ...): primary is a seagrass CC
      that didn't fully merge into any structural cluster. These represent
      "this is a productive seagrass area" rather than "this is a
      structure to fish at." Rendered with a teal-bordered bbox.

  Cross-reference: every anchor (structural or habitat) records
  `within_seagrass_bed_ids` — the IDs of any seagrass CCs whose pixel_bbox
  contains the anchor's primary center. So a structural creek-mouth anchor
  surrounded by a giant seagrass flat carries the bed's id even though the
  bed is its own habitat anchor.

  Tier thresholds:
    - SEAGRASS_BED_LARGE  (CC area >= 10000 px)
    - SEAGRASS_BED_MEDIUM ( 1500..10000 px )
    - SEAGRASS_BED_SMALL  (  200.. 1500 px )
    - OYSTER_BAR (CC area >= 30 px)

  Habitat masks are also rendered as semi-transparent tints on the base
  image so spatial extent is visible alongside the point candidates.

  Phase 3b (anchor-of-anchors): a "low value" anchor sitting close to a
  "high value" anchor is recorded as the high-value anchor's CHILD via
  cross-reference (no merge — both anchors stay first-class so the
  footprint cap is preserved).

  Low value = (1 primary + 0 secondary) OR (1 primary + 1 secondary
              where the secondary is a "small tier" feature: POINT_R13,
              POINT_R26, POCKET_R13, POCKET_R26, OYSTER_BAR,
              SEAGRASS_BED_SMALL).
  High value = anything not low value.
  Close = EITHER
            (a) primary-to-primary center distance <= PARENT_CENTER_DISTANCE_PX
                AND anchor-bbox edge-to-edge distance <= PARENT_EDGE_DISTANCE_PX
            (b) low-value primary center is inside (or within
                PARENT_NEARLY_IN_PX of) the high-value anchor's bbox
                (override rule for cases where the rich anchor's bbox is
                large and the centers are far apart but the lone anchor
                sits geographically inside the rich anchor's footprint).
  Greedy single-pass: each low-value anchor picks its single closest
  qualifying parent (no chains, no re-parenting).
  Habitat anchors may be either parent or child.

DEDUP MODEL (Phase 1)

Two candidates are "duplicates" iff:
  (1) their pixel_centers are within DEDUP_DISTANCE_PX of each other, AND
  (2) they belong to the same compatibility group.

Compatibility groups (features that COULD describe the same physical thing):
  Group W = narrow-water features
            (DRAIN, CREEK_MOUTH, LARGE_POCKET, SHOAL, POCKET_R13, POCKET_R26)
  Group L = narrow-land features (POINT_R13, POINT_R26)
  Group I = land bodies (ISLAND_SMALL, ISLAND_MEDIUM, ISLAND_LARGE)
  Group S = seagrass tiers (SEAGRASS_BED_LARGE/MEDIUM/SMALL) — three size
            tiers of the SAME physical bed CC are mutually exclusive, but in
            practice CCs are non-overlapping so dedup is a no-op here.
  Group O = oyster (OYSTER_BAR)

Cross-group pairs are NOT dedup candidates (water features and land features
can't be the same physical thing even if they're at the same pixel center,
and habitat features describe different cover-class entities).

Within a duplicate pair, the higher-priority category wins. CATEGORY_PRIORITY
defines the order. Priority is roughly "more comprehensive / more informative"
first; e.g. DRAIN > LARGE_POCKET (a drain says more than just "pocket exists")
and POINT_R26 > POINT_R13 (R26 catches the same shape with more pixels).

CLUSTERING MODEL (Phase 2)

  - Sort surviving candidates by bbox area DESCENDING (largest first;
    tie-breaker = CATEGORY_PRIORITY). This makes "more encompassing"
    features get to claim their cluster first.
  - Greedy: pop the next unassigned candidate as a primary. Find every
    OTHER unassigned candidate whose center is within SECONDARY_RADIUS_PX
    of the primary's bbox. Those become secondary; mark them assigned.
  - After all primaries are assigned, do a tertiary pass: for each anchor,
    list candidate ids that belong to OTHER anchors but whose centers sit
    within TERTIARY_RADIUS_PX of THIS anchor's primary center. These are
    "neighbor" features — informational only; they're owned by a different
    anchor. The same feature can be tertiary of multiple anchors.

  Each member candidate has exactly ONE home anchor where it's primary or
  secondary; tertiary is a non-exclusive nearby-features list.

Usage (via shim):
  python scripts/cv_detect_all.py --cell root-10-8
  python scripts/cv_detect_all.py --cell root-10-8 --cell root-11-5
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from readwater import storage
from readwater.pipeline.cv.helpers import (
    connected_components,
    grid_cell_for,
    load_font,
    load_habitat_mask,
)

# ---- Dedup parameters ----

DEDUP_DISTANCE_PX = 15        # centers within this many px = same physical feature

# ---- Clustering parameters ----

SECONDARY_RADIUS_PX = 75      # other-feature center within this distance of primary's
                              # bbox -> secondary of that anchor
TERTIARY_RADIUS_PX = 200      # primary-center to other-anchor-member-center within this
                              # -> tertiary reference (informational only)
ANCHOR_FOOTPRINT_CAP_PX = 528 # max width or height of an anchor's union bbox
                              # (3x3 grid cells = 480 px, + 10% slack ≈ 528 px).
                              # Attaching a secondary that pushes the cluster past
                              # this cap is rejected; the candidate stays free to
                              # become its own anchor.

# ---- Phase 3b parent/child parameters ----
# At z16 over Rookery Bay (~26°N), 1 px ≈ 2.15 m. The initial spec was
# center<=150m / edge<=25m (== 70px / 12px) but on real cells that fired
# zero links because almost no candidate-pair sits with both centers <70px
# AND bboxes touching to within 12px. Loosened by ~3x after viewing actual
# nearest-neighbor distances on root-10-8 + root-11-5.
PARENT_CENTER_DISTANCE_PX = 200  # ≈ 430 m: low-value primary must be within this
                                 # of a high-value primary to qualify as its child
PARENT_EDGE_DISTANCE_PX = 50     # ≈ 107 m: AND the two anchor bboxes must be within
                                 # this edge-to-edge distance (the tighter gate;
                                 # protects against center-only links across gaps)
# Override rule: a low-value anchor whose primary center sits inside (or
# within NEARLY_IN_PX of) a high-value anchor's bbox links regardless of
# center distance. Catches "small lone anchor sitting inside a big rich
# anchor's footprint" cases where the rich anchor's bbox is so large that
# the centers are far apart but they're geographically the same area.
PARENT_NEARLY_IN_PX = 25         # ≈  54 m

# Categories considered "small tier" for the low-value classifier — a
# feature that's just a small scale variant rather than a real structural
# feature. A 1-primary-1-secondary anchor still counts as low-value if the
# secondary is in this set.
SMALL_TIER_CATEGORIES = {
    "POINT_R13", "POINT_R26", "POCKET_R13", "POCKET_R26",
    "OYSTER_BAR", "SEAGRASS_BED_SMALL",
}

# Categories grouped by what they describe physically. Only intra-group pairs
# can be duplicates.
GROUP_WATER_NARROW = {
    "DRAIN", "CREEK_MOUTH", "LARGE_POCKET", "SHOAL",
    "POCKET_R13", "POCKET_R26",
}
GROUP_LAND_NARROW = {"POINT_R13", "POINT_R26"}
GROUP_LAND_BODY = {"ISLAND_SMALL", "ISLAND_MEDIUM", "ISLAND_LARGE"}
GROUP_SEAGRASS = {"SEAGRASS_BED_LARGE", "SEAGRASS_BED_MEDIUM", "SEAGRASS_BED_SMALL"}
GROUP_OYSTER = {"OYSTER_BAR"}


def category_group(category: str) -> str | None:
    if category in GROUP_WATER_NARROW: return "W"
    if category in GROUP_LAND_NARROW:  return "L"
    if category in GROUP_LAND_BODY:    return "I"
    if category in GROUP_SEAGRASS:     return "S"
    if category in GROUP_OYSTER:       return "O"
    return None


# Priority for dedup tie-breaking. Higher = preferred when categories collide
# at the same physical location. Roughly "more informative" first. Used by
# Phase 1 dedup only.
CATEGORY_PRIORITY = {
    # narrow water (drains beats pockets when they describe the same throat)
    "DRAIN":               100,
    "CREEK_MOUTH":          90,
    "LARGE_POCKET":         80,
    "SHOAL":                70,
    "POCKET_R26":           60,
    "POCKET_R13":           50,
    # land bodies (larger tier wins; in practice same island can't be in
    # two tiers, but list anyway for completeness)
    "ISLAND_LARGE":         85,
    "ISLAND_MEDIUM":        75,
    "ISLAND_SMALL":         55,
    # narrow land (R26 catches same shape with more pixels than R13)
    "POINT_R26":            45,
    "POINT_R13":            35,
    # habitat (purely for completeness — habitat CCs are non-overlapping
    # within a layer, so dedup pairs are vanishingly rare)
    "SEAGRASS_BED_LARGE":   65,
    "SEAGRASS_BED_MEDIUM":  45,
    "SEAGRASS_BED_SMALL":   25,
    "OYSTER_BAR":           20,
}

# Priority for choosing the PRIMARY of an anchor group in Phase 2 clustering.
# Different from CATEGORY_PRIORITY: islands are always primary when present in
# a cluster, because an island is a discrete physical entity while drains /
# pockets / points / shoals are characteristics OF an area. Within tiers,
# bbox area breaks ties.
CLUSTER_PRIMARY_PRIORITY = {
    # Tier 1: discrete land bodies — always preferred as primary when in cluster
    "ISLAND_LARGE":         100,
    "ISLAND_MEDIUM":         90,
    "ISLAND_SMALL":          80,
    # Tier 2: comprehensive water features
    "DRAIN":                 70,
    "CREEK_MOUTH":           65,
    "LARGE_POCKET":          60,
    "SHOAL":                 55,
    # Tier 3: detector-specific scale tiers (less comprehensive)
    "POINT_R26":             40,
    "POCKET_R26":            40,
    "POINT_R13":             30,
    "POCKET_R13":            30,
    # Tier 4: habitat — habitat is a cover class, not a structural feature.
    # Structural anchors always win primary status when present in a cluster.
    # A seagrass bed only becomes primary when nothing structural is in
    # range — that's the definition of "open seagrass flat with no structure"
    # and is exactly the case where it deserves to anchor on its own.
    "SEAGRASS_BED_LARGE":    20,
    "SEAGRASS_BED_MEDIUM":   18,
    "SEAGRASS_BED_SMALL":    15,
    "OYSTER_BAR":            10,
}


# ---- Per-category visual rendering (consistent across detectors) ----

CATEGORY_COLOR = {
    "DRAIN":               (220,  30,  30, 255),    # red
    "CREEK_MOUTH":         (255, 165,   0, 255),    # orange
    "LARGE_POCKET":        (255, 230,  50, 255),    # yellow
    "SHOAL":               ( 60, 200, 230, 255),    # cyan-ish
    "ISLAND_SMALL":        (150, 230, 100, 255),    # light green
    "ISLAND_MEDIUM":       ( 50, 180,  50, 255),    # medium green
    "ISLAND_LARGE":        ( 30, 110,  30, 255),    # dark green
    "POINT_R13":           (255, 150, 200, 255),    # pink
    "POINT_R26":           (220,  80, 180, 255),    # magenta
    "POCKET_R13":          (140, 200, 255, 255),    # light blue
    "POCKET_R26":          ( 30, 110, 200, 255),    # deep blue
    # Habitat — kept distinct from the (water/land) green family so seagrass
    # markers don't get confused with islands. Teal/turquoise = SAV;
    # tan/beige = oyster reef.
    "SEAGRASS_BED_LARGE":  ( 20, 150, 130, 255),    # deep teal
    "SEAGRASS_BED_MEDIUM": ( 80, 200, 170, 255),    # mid teal
    "SEAGRASS_BED_SMALL":  (160, 230, 210, 255),    # pale teal
    "OYSTER_BAR":          (220, 180, 100, 255),    # tan
}

# Tint colors used to fill the actual habitat extent on the satellite image
# (rendered before the dot overlay). Alpha is intentionally low so the base
# satellite is still readable.
SEAGRASS_TINT_RGBA = ( 40, 200, 170,  90)
OYSTER_TINT_RGBA   = (245, 220, 130, 220)   # higher alpha — oyster reefs are tiny


# ---- IO ----


def latest_json(area_id: str, cell_id: str, prefix: str) -> Path | None:
    """Return the newest <prefix>*.json file in the cell's structures dir, or None.

    Routes through ``storage.cell_structures_dir`` so the same code works
    for any area and is decoupled from the on-disk layout. Manifest-based
    lookup is a future refinement; for now glob-the-latest is fine.
    """
    d = storage.cell_structures_dir(area_id, cell_id)
    if not d.exists():
        return None
    matches = sorted(d.glob(f"{prefix}*.json"))
    return matches[-1] if matches else None


def load_detector(area_id: str, cell_id: str, prefix: str, source: str) -> list[dict]:
    """Load the latest detector JSON and normalize each candidate to a common
    record. Returns [] if the detector hasn't been run for this cell.
    """
    p = latest_json(area_id, cell_id, prefix)
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
            "pixel_bbox": list(c["pixel_bbox"]),
            "pixel_center": list(c["pixel_center"]),
            "extra": c,                             # preserve everything for downstream
        })
    return out


# ---- Habitat candidate detection (Phase 3a) ----

# Seagrass tier thresholds (pixels of CC area). Calibrated against root-10-8
# and root-11-5 — see scratch/diagnose_no_smooth.py / habitat CC scan.
SEAGRASS_LARGE_MIN = 10000
SEAGRASS_MEDIUM_MIN = 1500
SEAGRASS_SMALL_MIN = 200       # below this we drop the CC entirely
OYSTER_MIN = 30                # FWC oyster polygons are small; keep most


def _classify_seagrass(area: int) -> str | None:
    if area >= SEAGRASS_LARGE_MIN:  return "SEAGRASS_BED_LARGE"
    if area >= SEAGRASS_MEDIUM_MIN: return "SEAGRASS_BED_MEDIUM"
    if area >= SEAGRASS_SMALL_MIN:  return "SEAGRASS_BED_SMALL"
    return None


def _habitat_cc_to_candidate(cc: dict, category: str, idx: int,
                             source: str, kind: str) -> dict:
    """Convert a habitat CC to a candidate record matching the same shape as
    the detector JSON candidates.
    """
    cx, cy = cc["center"]
    x0, y0, x1, y1 = cc["bbox"]
    short = "s" if kind == "seagrass" else "o"
    return {
        "id": f"{short}{idx}",                       # s1, s2, ..., o1, o2
        "category": category,
        "source_detector": source,
        "pixel_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "pixel_center": [int(round(cx)), int(round(cy))],
        "extra": {
            "id": f"{short}{idx}",
            "category": category,
            "pixel_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "pixel_center": [int(round(cx)), int(round(cy))],
            "area_px": int(cc["area"]),
            "habitat_kind": kind,
        },
    }


def detect_habitat_candidates(area_id: str, cell_id: str
                              ) -> tuple[list[dict], list[dict],
                                         np.ndarray | None, np.ndarray | None,
                                         dict]:
    """Run inline habitat detection for a cell.

    Returns:
        (seagrass_candidates, oyster_candidates,
         seagrass_mask | None, oyster_mask | None,
         summary_dict)

    The masks are returned so the renderer can tint them onto the base image.
    summary_dict has the per-cell aggregate stats for the JSON output.
    """
    summary: dict = {}

    # --- Seagrass ---
    sea_mask = load_habitat_mask(area_id, cell_id, "seagrass")
    sea_cands: list[dict] = []
    if sea_mask is None:
        print(f"  [warn] {cell_id}: no seagrass mask found, skipping seagrass")
        summary["seagrass"] = None
    else:
        sea_ccs = connected_components(sea_mask, min_pixels=SEAGRASS_SMALL_MIN)
        sea_ccs.sort(key=lambda c: -c["area"])
        idx = 0
        for cc in sea_ccs:
            cat = _classify_seagrass(cc["area"])
            if cat is None:
                continue
            idx += 1
            sea_cands.append(_habitat_cc_to_candidate(cc, cat, idx, "habitat", "seagrass"))
        summary["seagrass"] = {
            "total_px": int(sea_mask.sum()),
            "pct_of_cell": float(sea_mask.mean()),
            "cc_count": len(sea_ccs),
            "candidate_count": len(sea_cands),
            "tier_counts": {
                "SEAGRASS_BED_LARGE":
                    sum(1 for c in sea_cands if c["category"] == "SEAGRASS_BED_LARGE"),
                "SEAGRASS_BED_MEDIUM":
                    sum(1 for c in sea_cands if c["category"] == "SEAGRASS_BED_MEDIUM"),
                "SEAGRASS_BED_SMALL":
                    sum(1 for c in sea_cands if c["category"] == "SEAGRASS_BED_SMALL"),
            },
        }

    # --- Oyster ---
    oys_mask = load_habitat_mask(area_id, cell_id, "oyster")
    oys_cands: list[dict] = []
    if oys_mask is None:
        print(f"  [warn] {cell_id}: no oyster mask found, skipping oyster")
        summary["oyster"] = None
    else:
        oys_ccs = connected_components(oys_mask, min_pixels=OYSTER_MIN)
        oys_ccs.sort(key=lambda c: -c["area"])
        for i, cc in enumerate(oys_ccs, start=1):
            oys_cands.append(_habitat_cc_to_candidate(cc, "OYSTER_BAR", i, "habitat", "oyster"))
        summary["oyster"] = {
            "total_px": int(oys_mask.sum()),
            "pct_of_cell": float(oys_mask.mean()),
            "cc_count": len(oys_ccs),
            "candidate_count": len(oys_cands),
        }

    return sea_cands, oys_cands, sea_mask, oys_mask, summary


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


# ---- Clustering ----


def _distance_point_to_bbox(point: list[float], bbox: list[int]) -> float:
    """Min Euclidean distance from a point to a bbox (0 if inside)."""
    x, y = point
    dx = max(bbox[0] - x, 0, x - bbox[2])
    dy = max(bbox[1] - y, 0, y - bbox[3])
    return (dx * dx + dy * dy) ** 0.5


def _euclidean(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _union_bbox(bboxes: list[list[int]]) -> list[int]:
    if not bboxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _bbox_within_cap(bbox: list[int], cap: int) -> bool:
    """True if both width and height of bbox fit inside the cap."""
    return (bbox[2] - bbox[0]) <= cap and (bbox[3] - bbox[1]) <= cap


def _category_is_habitat(category: str) -> bool:
    return category in GROUP_SEAGRASS or category in GROUP_OYSTER


def cluster_into_anchors(kept: list[dict],
                         secondary_radius: int = SECONDARY_RADIUS_PX,
                         tertiary_radius: int = TERTIARY_RADIUS_PX,
                         footprint_cap: int = ANCHOR_FOOTPRINT_CAP_PX
                         ) -> list[dict]:
    """Group surviving (post-dedup) candidates into anchor groups.

    Each candidate goes into exactly one anchor as either primary or secondary.
    Tertiary is recorded as a list of candidate IDs that belong to OTHER
    anchors but happen to be within tertiary_radius of this anchor's primary
    center — informational, non-exclusive.

    With CLUSTER_PRIMARY_PRIORITY now demoting habitat below all structural
    categories, the greedy primary-selection loop naturally produces all
    structural anchors first, then any unassigned habitat candidates form
    their own (habitat-flavored) anchors. We mark each anchor with a
    `type` field ("structural" or "habitat") based on its primary category.

    Footprint cap: when attaching a candidate as secondary would push the
    anchor's union bbox beyond `footprint_cap` on either axis, the candidate
    is rejected and remains free to become its own anchor downstream.

    Returns a list of anchor dicts, each with:
      id (a1, a2, ... for structural; h1, h2, ... for habitat),
      type ("structural" | "habitat"),
      primary, secondary (list), tertiary_refs (list of IDs),
      anchor_bbox (union of primary + secondary bboxes),
      within_seagrass_bed_ids (filled in by post-processing).
    """
    # Sort: highest CLUSTER_PRIMARY_PRIORITY first; tie-breaker = bbox area
    # (largest first).
    sorted_cands = sorted(
        kept,
        key=lambda c: (-CLUSTER_PRIMARY_PRIORITY.get(c["category"], 0),
                       -_bbox_area(c["pixel_bbox"])),
    )

    assigned: set[str] = set()
    anchors: list[dict] = []
    n_struct = 0
    n_habitat = 0

    for primary in sorted_cands:
        if primary["id"] in assigned:
            continue

        is_habitat_primary = _category_is_habitat(primary["category"])
        primary_bbox = list(primary["pixel_bbox"])

        # Build a list of in-range candidates with their distances to the
        # primary bbox; sort ascending so the closest fit attempts first.
        # This matters now that the cap can reject later attachments.
        in_range: list[tuple[float, dict]] = []
        for other in kept:
            if other["id"] in assigned or other["id"] == primary["id"]:
                continue
            d = _distance_point_to_bbox(other["pixel_center"], primary_bbox)
            if d <= secondary_radius:
                in_range.append((d, other))
        in_range.sort(key=lambda t: t[0])

        secondary: list[dict] = []
        running_bbox = list(primary_bbox)
        for _d, other in in_range:
            candidate_union = _union_bbox([running_bbox, other["pixel_bbox"]])
            if not _bbox_within_cap(candidate_union, footprint_cap):
                # Attaching this would push the anchor over the size cap;
                # leave it unassigned so it can become its own anchor.
                continue
            secondary.append(other)
            assigned.add(other["id"])
            running_bbox = candidate_union

        assigned.add(primary["id"])
        if is_habitat_primary:
            n_habitat += 1
            anchor_id = f"h{n_habitat}"
            anchor_type = "habitat"
        else:
            n_struct += 1
            anchor_id = f"a{n_struct}"
            anchor_type = "structural"

        anchors.append({
            "id": anchor_id,
            "type": anchor_type,
            "primary": primary,
            "secondary": secondary,
            "anchor_bbox": running_bbox,
            "tertiary_refs": [],         # filled in below
            "within_seagrass_bed_ids": [],   # filled in below
        })

    # Tertiary pass: features in OTHER anchors within tertiary_radius of THIS
    # anchor's primary center (non-exclusive — same feature can be tertiary
    # of multiple anchors).
    for anchor in anchors:
        member_ids = {anchor["primary"]["id"]} | {s["id"] for s in anchor["secondary"]}
        primary_center = anchor["primary"]["pixel_center"]
        for other in kept:
            if other["id"] in member_ids:
                continue
            if _euclidean(primary_center, other["pixel_center"]) <= tertiary_radius:
                anchor["tertiary_refs"].append(other["id"])

    # Within-seagrass cross-reference pass: for every seagrass CC in `kept`,
    # find every STRUCTURAL anchor whose primary center sits inside that
    # seagrass CC's bbox and tag the anchor with the bed id. Habitat anchors
    # don't get this flag — "seagrass inside seagrass" is meaningless and
    # just creates noise from overlapping bboxes of disjoint CCs.
    seagrass_records = [c for c in kept if c["category"] in GROUP_SEAGRASS]
    for anchor in anchors:
        if anchor["type"] != "structural":
            continue
        pcx, pcy = anchor["primary"]["pixel_center"]
        for sg in seagrass_records:
            x0, y0, x1, y1 = sg["pixel_bbox"]
            if x0 <= pcx <= x1 and y0 <= pcy <= y1:
                anchor["within_seagrass_bed_ids"].append(sg["id"])

    return anchors


# ---- Phase 3b: anchor-of-anchors parent/child links ----


def _bbox_to_bbox_distance(a: list[int], b: list[int]) -> float:
    """Minimum Euclidean distance between two axis-aligned bboxes (0 if they
    overlap or touch).
    """
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def _is_low_value_anchor(anchor: dict) -> bool:
    """A low-value anchor has either no secondaries, or a single small-tier
    secondary. See SMALL_TIER_CATEGORIES.
    """
    secondary = anchor.get("secondary", [])
    if len(secondary) == 0:
        return True
    if len(secondary) == 1 and secondary[0]["category"] in SMALL_TIER_CATEGORIES:
        return True
    return False


def link_parent_child(anchors: list[dict],
                      center_distance: int = PARENT_CENTER_DISTANCE_PX,
                      edge_distance: int = PARENT_EDGE_DISTANCE_PX,
                      nearly_in: int = PARENT_NEARLY_IN_PX) -> None:
    """Mutate each anchor in place, adding `parent_anchor_id` (str | None)
    and `child_anchor_ids` (list[str]).

    Greedy single pass: every low-value anchor picks its single closest
    qualifying high-value parent. A pair qualifies if EITHER:
      (A) primary-to-primary center distance <= center_distance
          AND anchor-bbox edge-to-edge distance <= edge_distance
      (B) low-value primary center is inside (or within nearly_in of)
          the high-value anchor's bbox. This overrides the center-distance
          cap and catches cases where the rich anchor's bbox is so large
          that their centers are far apart but the lone low-value anchor
          sits geographically inside the rich anchor's footprint.

    Tie-breaker between qualifying parents: smallest center-to-center
    distance wins.

    No chains, no re-parenting: a low-value anchor that becomes someone's
    child is not subsequently treated as a parent. Habitat anchors may be
    either parent or child.
    """
    for a in anchors:
        a.setdefault("parent_anchor_id", None)
        a.setdefault("child_anchor_ids", [])

    low = [a for a in anchors if _is_low_value_anchor(a)]
    high = [a for a in anchors if not _is_low_value_anchor(a)]

    for child in low:
        cpc = child["primary"]["pixel_center"]
        cbb = child["anchor_bbox"]
        best_parent: dict | None = None
        best_d = float("inf")
        for parent in high:
            if parent["id"] == child["id"]:
                continue
            ppc = parent["primary"]["pixel_center"]
            pbb = parent["anchor_bbox"]
            center_d = _euclidean(cpc, ppc)
            edge_d = _bbox_to_bbox_distance(cbb, pbb)
            in_d = _distance_point_to_bbox(cpc, pbb)
            qualifies = (
                (center_d <= center_distance and edge_d <= edge_distance)
                or in_d <= nearly_in
            )
            if not qualifies:
                continue
            if center_d < best_d:
                best_d = center_d
                best_parent = parent
        if best_parent is not None:
            child["parent_anchor_id"] = best_parent["id"]
            best_parent["child_anchor_ids"].append(child["id"])


# ---- Rendering ----


LEGEND_PANEL_HEIGHT = 480     # px below the satellite for the off-image legend
LEGEND_FONT_SIZE = 13


def _tint_layer(mask: np.ndarray, rgba: tuple[int, int, int, int],
                size: tuple[int, int]) -> Image.Image:
    """Build an RGBA image of `size` with the given color where mask is True
    and full transparency elsewhere.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if mask is None or not mask.any():
        return layer
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[mask, 0] = rgba[0]
    arr[mask, 1] = rgba[1]
    arr[mask, 2] = rgba[2]
    arr[mask, 3] = rgba[3]
    return Image.fromarray(arr, mode="RGBA")


def render_combined_overlay(base_image_path: Path,
                            anchors: list[dict],
                            dropped: list[dict],
                            output_path: Path,
                            image_size: tuple[int, int] = (1280, 1280),
                            seagrass_mask: np.ndarray | None = None,
                            oyster_mask: np.ndarray | None = None,
                            habitat_summary: dict | None = None) -> Path:
    base = Image.open(base_image_path).convert("RGBA")
    canvas = Image.new("RGBA", (image_size[0], image_size[1] + LEGEND_PANEL_HEIGHT),
                       (20, 20, 20, 255))

    # ---- Habitat tints (composited under the candidate dots) ----
    if seagrass_mask is not None:
        sea_layer = _tint_layer(seagrass_mask, SEAGRASS_TINT_RGBA, base.size)
        base = Image.alpha_composite(base, sea_layer)
    if oyster_mask is not None:
        oys_layer = _tint_layer(oyster_mask, OYSTER_TINT_RGBA, base.size)
        base = Image.alpha_composite(base, oys_layer)

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

    label_font = load_font(13)
    primary_label_font = load_font(16)

    # Pass 1: draw each anchor's bounding rectangle. Structural anchors get
    # a faint white outline; habitat anchors get a teal outline so they're
    # visually distinct from "this is a place to fish at" structural anchors.
    for anchor in anchors:
        ax0, ay0, ax1, ay1 = anchor["anchor_bbox"]
        # Pad a few pixels so the box visually contains the dots
        ax0 -= 6; ay0 -= 6; ax1 += 6; ay1 += 6
        if anchor.get("type") == "habitat":
            outline = (40, 200, 170, 200)   # bright teal
            width = 2
        else:
            outline = (220, 220, 220, 140)  # faint white-grey
            width = 1
        draw.rectangle([ax0, ay0, ax1, ay1], outline=outline, width=width)

    # Pass 1b: draw parent<->child connecting lines (Phase 3b). Goes under
    # the dots so the primary markers sit on top of the line endpoints.
    by_id = {a["id"]: a for a in anchors}
    for child in anchors:
        parent_id = child.get("parent_anchor_id")
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        cx, cy = child["primary"]["pixel_center"]
        px, py = parent["primary"]["pixel_center"]
        # Black under-stroke for contrast on light backgrounds, then a yellow
        # stroke on top so it pops against the satellite + tint.
        draw.line([(cx, cy), (px, py)], fill=(0, 0, 0, 220), width=4)
        draw.line([(cx, cy), (px, py)], fill=(255, 220, 80, 230), width=2)

    # Pass 2: draw secondary dots (so primaries render on top of them).
    for anchor in anchors:
        for sec in anchor["secondary"]:
            cx, cy = sec["pixel_center"]
            color = CATEGORY_COLOR.get(sec["category"], (200, 200, 200, 255))
            r = 5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=color, outline=(0, 0, 0, 255), width=1)
            text = sec["id"]
            tx = int(cx) + 7; ty = int(cy) - 12
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), text, fill="black", font=label_font)
            draw.text((tx, ty), text, fill=color, font=label_font)

    # Pass 3: draw primaries (larger dot + anchor id like a1).
    for anchor in anchors:
        primary = anchor["primary"]
        cx, cy = primary["pixel_center"]
        color = CATEGORY_COLOR.get(primary["category"], (200, 200, 200, 255))
        r = 9
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=color, outline=(255, 255, 255, 255), width=2)
        text = anchor["id"].upper()
        tx = int(cx) + 11; ty = int(cy) - 16
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), text,
                              fill="black", font=primary_label_font)
        draw.text((tx, ty), text, fill=color, font=primary_label_font)

    composed_top = Image.alpha_composite(base, layer)
    canvas.paste(composed_top, (0, 0))

    # ---- Off-satellite legend panel ----
    legend_draw = ImageDraw.Draw(canvas)
    panel_y0 = image_size[1]
    legend_font = load_font(LEGEND_FONT_SIZE)
    title_font = load_font(15)

    n_kept = sum(1 + len(a["secondary"]) for a in anchors)
    n_struct = sum(1 for a in anchors if a.get("type") == "structural")
    n_habitat = sum(1 for a in anchors if a.get("type") == "habitat")
    n_children = sum(1 for a in anchors if a.get("parent_anchor_id") is not None)
    n_parents = sum(1 for a in anchors if a.get("child_anchor_ids"))
    legend_draw.text(
        (10, panel_y0 + 8),
        f"Phase 3b: {n_struct} structural + {n_habitat} habitat anchors  |  "
        f"{n_kept} member features, {len(dropped)} dedup-dropped  |  "
        f"{n_children} child links to {n_parents} parents (yellow lines)  |  "
        f"sec={SECONDARY_RADIUS_PX}px tert={TERTIARY_RADIUS_PX}px "
        f"cap={ANCHOR_FOOTPRINT_CAP_PX}px "
        f"parent={PARENT_CENTER_DISTANCE_PX}/{PARENT_EDGE_DISTANCE_PX}/{PARENT_NEARLY_IN_PX}px",
        fill=(255, 255, 255, 255), font=title_font,
    )

    line_h = LEGEND_FONT_SIZE + 4
    if habitat_summary:
        sea = habitat_summary.get("seagrass") or {}
        oys = habitat_summary.get("oyster") or {}
        sea_pct = sea.get("pct_of_cell", 0.0) * 100
        oys_pct = oys.get("pct_of_cell", 0.0) * 100
        tier = sea.get("tier_counts", {}) or {}
        sea_msg = (
            f"seagrass {sea_pct:5.2f}% of cell, "
            f"{sea.get('candidate_count', 0)} beds "
            f"(L={tier.get('SEAGRASS_BED_LARGE', 0)} "
            f"M={tier.get('SEAGRASS_BED_MEDIUM', 0)} "
            f"S={tier.get('SEAGRASS_BED_SMALL', 0)})"
        ) if sea else "seagrass: missing"
        oys_msg = (
            f"oyster {oys_pct:5.2f}% of cell, "
            f"{oys.get('candidate_count', 0)} bars"
        ) if oys else "oyster: missing"
        legend_draw.text(
            (10, panel_y0 + 8 + (LEGEND_FONT_SIZE + 4)),
            f"Habitat: {sea_msg}   |   {oys_msg}",
            fill=(180, 230, 220, 255), font=legend_font,
        )

    header_y = panel_y0 + 32 + (LEGEND_FONT_SIZE + 4 if habitat_summary else 0)
    legend_draw.text((10, header_y),
                     "ANCHORS  (primary [secondary count] tertiary count)",
                     fill=(180, 255, 180, 255), font=title_font)

    y = header_y + line_h + 4
    max_lines = (LEGEND_PANEL_HEIGHT - (y - panel_y0) - 4) // line_h
    for anchor in anchors[:max_lines]:
        primary = anchor["primary"]
        color = CATEGORY_COLOR.get(primary["category"], (200, 200, 200, 255))
        pcx, pcy = primary["pixel_center"]
        cell = grid_cell_for(pcx, pcy, image_size)
        legend_draw.rectangle([10, y + 2, 22, y + LEGEND_FONT_SIZE], fill=color)
        sec_summary = ", ".join(
            f"{s['id']}/{s['category']}" for s in anchor["secondary"]
        )
        if not sec_summary:
            sec_summary = "—"
        if len(sec_summary) > 70:
            sec_summary = sec_summary[:67] + "..."
        within = anchor.get("within_seagrass_bed_ids") or []
        within_str = f" inSG=[{','.join(within)}]" if within else ""
        parent_id = anchor.get("parent_anchor_id")
        children = anchor.get("child_anchor_ids") or []
        rel_str = ""
        if parent_id:
            rel_str = f"  -> parent={parent_id.upper()}"
        elif children:
            rel_str = f"  <- children=[{','.join(c.upper() for c in children)}]"
        line = (
            f"{anchor['id'].upper():<3s} "
            f"{primary['id']:>4s} {primary['category']:<14s} @{cell:<3s} "
            f"sec[{len(anchor['secondary'])}]={sec_summary}  "
            f"tert={len(anchor['tertiary_refs'])}{within_str}{rel_str}"
        )
        # Color-coding:
        #   - habitat anchor: teal
        #   - structural child of someone: dim yellow (matches link line)
        #   - structural parent of children: bright yellow
        #   - structural standalone: white
        if anchor.get("type") == "habitat":
            text_color = (180, 240, 220, 255)
        elif parent_id:
            text_color = (220, 200, 130, 255)
        elif children:
            text_color = (255, 230, 100, 255)
        else:
            text_color = (255, 255, 255, 255)
        legend_draw.text((28, y), line, fill=text_color, font=legend_font)
        y += line_h
    if len(anchors) > max_lines:
        legend_draw.text((28, y),
                         f"... +{len(anchors) - max_lines} more anchors (see JSON)",
                         fill=(180, 180, 180, 255), font=legend_font)

    composed = canvas.convert("RGB")
    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    storage.atomic_write_bytes(output_path, buf.getvalue())
    return output_path


# ---- CLI ----


PHASE = "3b"
SCHEMA_VERSION = PHASE  # orchestrator's schema_version mirrors the phase tag


def run_one(area_id: str, cell_id: str) -> int:
    z16_path = storage.z16_image_path(area_id, cell_id)
    out_dir = storage.cell_structures_dir(area_id, cell_id)

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
        loaded = load_detector(area_id, cell_id, prefix, source)
        per_source_counts[source] = len(loaded)
        candidates.extend(loaded)

    # Phase 3a: detect habitat candidates inline (no separate detector script)
    sea_cands, oys_cands, sea_mask, oys_mask, habitat_summary = (
        detect_habitat_candidates(area_id, cell_id)
    )
    per_source_counts["habitat-sea"] = len(sea_cands)
    per_source_counts["habitat-oys"] = len(oys_cands)
    candidates.extend(sea_cands)
    candidates.extend(oys_cands)

    print(f"  loaded:  " + "  ".join(f"{s}={n}" for s, n in per_source_counts.items())
          + f"  -> {len(candidates)} total")

    kept, dropped = dedup_candidates(candidates)
    print(f"  dedup:   {len(kept)} kept,  {len(dropped)} dropped "
          f"(threshold {DEDUP_DISTANCE_PX} px)")

    if dropped:
        print(f"  Dropped duplicates (loser -> winner):")
        for d in dropped:
            w = d["dropped_for"]
            print(f"    {d['source_detector'][:7]:<7s} {d['id']:>4s} {d['category']:<14s}"
                  f"  ->  {w['winner_source'][:7]:<7s} {w['winner_id']:>4s} {w['winner_category']}")

    # Phase 2: cluster surviving candidates into anchor groups
    anchors = cluster_into_anchors(kept,
                                    secondary_radius=SECONDARY_RADIUS_PX,
                                    tertiary_radius=TERTIARY_RADIUS_PX)
    n_secondary = sum(len(a["secondary"]) for a in anchors)
    n_struct = sum(1 for a in anchors if a.get("type") == "structural")
    n_habitat = sum(1 for a in anchors if a.get("type") == "habitat")
    n_with_sg = sum(1 for a in anchors if a.get("within_seagrass_bed_ids"))
    print(f"  cluster: {len(anchors)} anchors  "
          f"({n_struct} structural + {n_habitat} habitat, "
          f"secondary={n_secondary}, "
          f"{n_with_sg} anchors flagged within_seagrass_bed)")

    # Phase 3b: parent/child links between low-value and high-value anchors
    link_parent_child(anchors,
                      center_distance=PARENT_CENTER_DISTANCE_PX,
                      edge_distance=PARENT_EDGE_DISTANCE_PX)
    n_children = sum(1 for a in anchors if a.get("parent_anchor_id") is not None)
    n_parents = sum(1 for a in anchors if a.get("child_anchor_ids"))
    n_low = sum(1 for a in anchors if _is_low_value_anchor(a))
    print(f"  link3b: {n_low} low-value anchors -> "
          f"{n_children} linked as children to {n_parents} parents "
          f"(center<={PARENT_CENTER_DISTANCE_PX}px AND edge<={PARENT_EDGE_DISTANCE_PX}px, "
          f"or low-center within {PARENT_NEARLY_IN_PX}px of high-bbox)")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    overlay_path = storage.cell_artifact_path(area_id, cell_id, "all", ts, "png")
    json_path = storage.cell_artifact_path(area_id, cell_id, "all", ts, "json")

    render_combined_overlay(z16_path, anchors, dropped, overlay_path,
                            seagrass_mask=sea_mask,
                            oyster_mask=oys_mask,
                            habitat_summary=habitat_summary)

    # JSON: medium-with-pointers per anchor; full source records in `members`
    def _light_record(c: dict) -> dict:
        return {
            "id": c["id"],
            "category": c["category"],
            "source_detector": c["source_detector"],
            "pixel_center": c["pixel_center"],
            "pixel_bbox": c["pixel_bbox"],
        }

    members_blob: dict[str, dict] = {}
    for a in anchors:
        for m in [a["primary"]] + a["secondary"]:
            members_blob[m["id"]] = m["extra"]

    storage.atomic_write_json(json_path, {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "phase": PHASE,
        "dedup_distance_px": DEDUP_DISTANCE_PX,
        "secondary_radius_px": SECONDARY_RADIUS_PX,
        "tertiary_radius_px": TERTIARY_RADIUS_PX,
        "anchor_footprint_cap_px": ANCHOR_FOOTPRINT_CAP_PX,
        "parent_center_distance_px": PARENT_CENTER_DISTANCE_PX,
        "parent_edge_distance_px": PARENT_EDGE_DISTANCE_PX,
        "parent_nearly_in_px": PARENT_NEARLY_IN_PX,
        "loaded_per_source": per_source_counts,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "anchor_count": len(anchors),
        "structural_anchor_count": sum(1 for a in anchors if a.get("type") == "structural"),
        "habitat_anchor_count": sum(1 for a in anchors if a.get("type") == "habitat"),
        "parent_anchor_count": sum(1 for a in anchors if a.get("child_anchor_ids")),
        "child_anchor_count": sum(1 for a in anchors if a.get("parent_anchor_id") is not None),
        "habitat_summary": habitat_summary,
        "anchors": [
            {
                "id": a["id"],
                "type": a.get("type", "structural"),
                "anchor_bbox": a["anchor_bbox"],
                "primary": _light_record(a["primary"]),
                "secondary": [_light_record(s) for s in a["secondary"]],
                "tertiary_refs": a["tertiary_refs"],
                "within_seagrass_bed_ids": a.get("within_seagrass_bed_ids", []),
                "parent_anchor_id": a.get("parent_anchor_id"),
                "child_anchor_ids": a.get("child_anchor_ids", []),
            }
            for a in anchors
        ],
        "members": members_blob,
        "dropped": dropped,
    })

    print(f"  overlay: {overlay_path.relative_to(storage.data_root().parent)}")
    print(f"  json:    {json_path.relative_to(storage.data_root().parent)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", default="rookery_bay_v2",
                        help="Area id (default: rookery_bay_v2).")
    parser.add_argument("--cell", action="append", required=True,
                        help="Cell id like root-10-8. Repeat for multiple.")
    args = parser.parse_args()
    rc_overall = 0
    for cell_id in args.cell:
        rc = run_one(args.area, cell_id)
        if rc != 0:
            rc_overall = rc
    return rc_overall


if __name__ == "__main__":
    sys.exit(main())
