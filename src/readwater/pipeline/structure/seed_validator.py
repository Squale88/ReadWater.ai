"""Seed-point quality validator.

Runs between Claude's IDENTIFY output and the geometry extractor. Per feature,
checks a handful of sanity conditions. Returns one of three verdicts:

    PASS        — seeds look sane, proceed to extract
    REGENERATE  — seeds have a fixable issue; ask Claude for new ones once
    DROP        — seeds are unrecoverable; skip the feature and log it

The agent is responsible for enforcing the one-retry cap.

The validator is pure: no I/O, no LLM calls. Given seeds + image dims + mode,
it returns a verdict and a reason string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    PASS = "pass"
    REGENERATE = "regenerate"
    DROP = "drop"


@dataclass
class ValidationResult:
    verdict: Verdict
    reason: str = ""


# Tuning constants (fractions of image shorter dim).
MIN_POS_NEG_SEPARATION_FRAC = 0.05
MIN_POSITIVE_SPREAD_FRAC = 0.03


def validate_seeds(
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    image_size: tuple[int, int],  # (width, height)
    extraction_mode: str,         # "region" | "corridor" | "point_feature" | "edge_band"
    excluded_regions: list[tuple[int, int, int, int]] | None = None,
) -> ValidationResult:
    """Return a verdict for a feature's seed set.

    Args:
        positive_points: points Claude said are inside the feature.
        negative_points: points Claude said are nearby but outside.
        image_size: (width, height) of the source image in pixels.
        extraction_mode: extractor mode chosen for this feature. Affects the
            minimum-seeds check.
        excluded_regions: optional (x, y, w, h) rectangles where seeds should
            not land (legend area, watermark bands). Seeds inside any excluded
            region trigger REGENERATE.
    """
    w, h = image_size
    short = min(w, h)

    # 1. Minimum seeds per mode.
    min_required = _min_positives_for_mode(extraction_mode)
    if len(positive_points) < min_required:
        return ValidationResult(
            Verdict.REGENERATE,
            f"{extraction_mode} mode needs at least {min_required} positive "
            f"points, got {len(positive_points)}",
        )

    # 2. In-bounds.
    for i, (x, y) in enumerate(positive_points):
        if not (0 <= x < w and 0 <= y < h):
            return ValidationResult(
                Verdict.REGENERATE,
                f"positive point {i} ({x},{y}) out of image bounds {w}x{h}",
            )
    for i, (x, y) in enumerate(negative_points):
        if not (0 <= x < w and 0 <= y < h):
            return ValidationResult(
                Verdict.REGENERATE,
                f"negative point {i} ({x},{y}) out of image bounds {w}x{h}",
            )

    # 3. Positive/negative separation.
    min_separation = MIN_POS_NEG_SEPARATION_FRAC * short
    for i, p in enumerate(positive_points):
        for j, n in enumerate(negative_points):
            d = math.hypot(p[0] - n[0], p[1] - n[1])
            if d < min_separation:
                return ValidationResult(
                    Verdict.REGENERATE,
                    f"positive {i} and negative {j} are {d:.0f}px apart; "
                    f"need at least {min_separation:.0f}px",
                )

    # 4. Positive clustering (only matters when we have ≥3).
    if len(positive_points) >= 3:
        xs = [p[0] for p in positive_points]
        ys = [p[1] for p in positive_points]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        if spread < MIN_POSITIVE_SPREAD_FRAC * short:
            return ValidationResult(
                Verdict.REGENERATE,
                f"3+ positive points clustered within {spread}px; "
                f"expected spread >= {MIN_POSITIVE_SPREAD_FRAC * short:.0f}px",
            )

    # 5. Excluded regions (legend, watermarks).
    if excluded_regions:
        for label, point_list in (("positive", positive_points), ("negative", negative_points)):
            for i, (x, y) in enumerate(point_list):
                for rx, ry, rw, rh in excluded_regions:
                    if rx <= x < rx + rw and ry <= y < ry + rh:
                        return ValidationResult(
                            Verdict.REGENERATE,
                            f"{label} point {i} at ({x},{y}) landed inside an "
                            f"overlay region ({rx},{ry},{rw},{rh})",
                        )

    return ValidationResult(Verdict.PASS, "")


def _min_positives_for_mode(mode: str) -> int:
    if mode == "corridor":
        return 2
    if mode == "edge_band":
        return 2
    return 1


def default_excluded_regions(
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Default set of image regions to avoid: the bottom-left legend drawn by
    render_annotated(), and a small band along tile boundaries where Google
    watermarks live. Agent passes these to validate_seeds().
    """
    w, h = image_size
    regions: list[tuple[int, int, int, int]] = []

    # Legend box in the bottom-left corner drawn by render_annotated.
    # See mosaic.render_annotated — legend_w=220, legend_h varies. Add generous padding.
    legend_w = 240
    legend_h = 140
    pad = 16
    regions.append((pad, h - legend_h - pad, legend_w, legend_h))

    return regions
