"""Tests for the cell-based seed validator."""

from __future__ import annotations

from readwater.pipeline.structure.seed_validator import (
    Verdict,
    validate_cells,
)


def _run(cells, mode="region", rows=8, cols=8):
    return validate_cells(cells, rows, cols, extraction_mode=mode)


# --- PASS cases ---


def test_pass_region_one_cell():
    r = _run(["D4"], mode="region")
    assert r.verdict == Verdict.PASS
    assert r.parsed_cells == [(3, 3)]


def test_pass_region_multiple_cells():
    r = _run(["A1", "A2", "B1", "B2"], mode="region")
    assert r.verdict == Verdict.PASS
    assert len(r.parsed_cells) == 4


def test_pass_corridor_two_cells():
    r = _run(["C3", "E5"], mode="corridor")
    assert r.verdict == Verdict.PASS


def test_pass_edge_band_two_cells():
    r = _run(["C3", "D4"], mode="edge_band")
    assert r.verdict == Verdict.PASS


def test_pass_point_feature_one_cell():
    r = _run(["E5"], mode="point_feature")
    assert r.verdict == Verdict.PASS


# --- Minimum cells per mode ---


def test_corridor_one_cell_regenerates():
    r = _run(["D4"], mode="corridor")
    assert r.verdict == Verdict.REGENERATE
    assert "corridor" in r.reason


def test_edge_band_one_cell_regenerates():
    r = _run(["D4"], mode="edge_band")
    assert r.verdict == Verdict.REGENERATE


def test_region_zero_cells_regenerates():
    r = _run([], mode="region")
    assert r.verdict == Verdict.REGENERATE


# --- Invalid labels ---


def test_invalid_label_regenerates():
    r = _run(["junk", "D4"], mode="region")
    assert r.verdict == Verdict.REGENERATE
    assert "invalid" in r.reason.lower()


def test_numeric_first_label_invalid():
    r = _run(["1A"], mode="region")
    assert r.verdict == Verdict.REGENERATE


# --- Out of bounds ---


def test_out_of_bounds_regenerates():
    r = _run(["I1"], mode="region", rows=8, cols=8)  # I = row 8, past grid
    assert r.verdict == Verdict.REGENERATE
    assert "outside" in r.reason.lower()


def test_out_of_bounds_column_regenerates():
    r = _run(["A9"], mode="region", rows=8, cols=8)  # col 9, past 8
    assert r.verdict == Verdict.REGENERATE


# --- Duplicates silently deduplicated ---


def test_duplicates_are_deduplicated_and_pass():
    r = _run(["A1", "A1", "B2"], mode="region")
    assert r.verdict == Verdict.PASS
    assert len(r.parsed_cells) == 2


# --- Non-list input is a DROP ---


def test_non_list_cells_drops():
    r = validate_cells("hull_of_anchor", 8, 8, "region")
    assert r.verdict == Verdict.DROP


# --- After dedup, must still meet minimum ---


def test_corridor_with_duplicate_positives_below_min_regenerates():
    r = _run(["C3", "C3"], mode="corridor")  # dedup -> 1 cell, corridor needs 2
    assert r.verdict == Verdict.REGENERATE
