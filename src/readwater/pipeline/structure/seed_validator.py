"""Seed-quality validator (cells edition).

Runs between Claude's cell-list output and the geometry extractor. Per
feature, checks a handful of sanity conditions on the declared grid cells.
Returns one of three verdicts:

    PASS        — cells look sane, proceed to extract
    REGENERATE  — cells have a fixable issue; ask Claude once more
    DROP        — cells are unrecoverable; skip the feature and log it

The agent enforces the one-retry cap; the validator itself is pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from readwater.pipeline.structure.grid_overlay import parse_cell


class Verdict(str, Enum):
    PASS = "pass"
    REGENERATE = "regenerate"
    DROP = "drop"


@dataclass
class ValidationResult:
    verdict: Verdict
    reason: str = ""
    parsed_cells: list[tuple[int, int]] | None = None  # (row, col) pairs that parsed ok


def validate_cells(
    cell_labels: list[str],
    grid_rows: int,
    grid_cols: int,
    extraction_mode: str,
) -> ValidationResult:
    """Return a verdict for a feature's cell list.

    Args:
        cell_labels: labels Claude returned (e.g. ["C3", "C4"]).
        grid_rows, grid_cols: the grid dimensions Claude saw.
        extraction_mode: "region" | "corridor" | "point_feature" | "edge_band".
            Affects min-cell-count.
    """
    if not isinstance(cell_labels, list):
        return ValidationResult(Verdict.DROP, "cells field is not a list")

    # Minimum cells per mode.
    min_required = _min_cells_for_mode(extraction_mode)
    if len(cell_labels) < min_required:
        return ValidationResult(
            Verdict.REGENERATE,
            f"{extraction_mode} mode needs at least {min_required} cells, "
            f"got {len(cell_labels)}",
        )

    parsed: list[tuple[int, int]] = []
    invalid: list[str] = []
    out_of_bounds: list[str] = []
    seen: set[tuple[int, int]] = set()
    duplicates: list[str] = []

    for label in cell_labels:
        rc = parse_cell(label)
        if rc is None:
            invalid.append(str(label))
            continue
        row, col = rc
        if not (0 <= row < grid_rows and 0 <= col < grid_cols):
            out_of_bounds.append(label)
            continue
        if rc in seen:
            duplicates.append(label)
            continue
        seen.add(rc)
        parsed.append(rc)

    if invalid:
        return ValidationResult(
            Verdict.REGENERATE,
            f"invalid cell labels: {invalid[:5]}{'...' if len(invalid) > 5 else ''}",
        )
    if out_of_bounds:
        return ValidationResult(
            Verdict.REGENERATE,
            f"cell labels outside {grid_rows}x{grid_cols} grid: "
            f"{out_of_bounds[:5]}{'...' if len(out_of_bounds) > 5 else ''}",
        )
    # Duplicates aren't fatal; just dedup silently. No regenerate.
    if len(parsed) < min_required:
        return ValidationResult(
            Verdict.REGENERATE,
            f"only {len(parsed)} valid cells after dedup; need at least {min_required}",
        )

    return ValidationResult(Verdict.PASS, "", parsed_cells=parsed)


def _min_cells_for_mode(mode: str) -> int:
    if mode in ("corridor", "edge_band"):
        return 2
    return 1
