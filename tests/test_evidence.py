"""Tests for the evidence builder (per-cell coverage + prompt formatting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from readwater.pipeline.evidence import (
    build_evidence_table,
    compute_cell_coverage,
    format_evidence_for_prompt,
)


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    """Save a boolean mask as an 8-bit PNG (255 = True)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)
    return path


# --- compute_cell_coverage ---


def test_all_water_mask_yields_100_percent_everywhere(tmp_path: Path):
    mask = np.ones((1280, 1280), dtype=bool)
    p = _write_mask(tmp_path / "water.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    assert len(cov) == 64
    for cell, frac in cov.items():
        assert frac == 1.0, f"{cell} should be 100%, got {frac}"


def test_no_coverage_mask_yields_zero_everywhere(tmp_path: Path):
    mask = np.zeros((1280, 1280), dtype=bool)
    p = _write_mask(tmp_path / "empty.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    for frac in cov.values():
        assert frac == 0.0


def test_half_covered_mask_tags_correct_cells(tmp_path: Path):
    """Left half covered — cells in cols 1-4 should be 100%, cols 5-8 should be 0%."""
    mask = np.zeros((1280, 1280), dtype=bool)
    mask[:, :640] = True
    p = _write_mask(tmp_path / "left.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    for row_letter in "ABCDEFGH":
        for col in range(1, 5):
            assert cov[f"{row_letter}{col}"] == 1.0, f"{row_letter}{col} should be 100%"
        for col in range(5, 9):
            assert cov[f"{row_letter}{col}"] == 0.0, f"{row_letter}{col} should be 0%"


def test_single_cell_coverage_block(tmp_path: Path):
    """A 160x160 block at (0, 0) should fill cell A1 only."""
    mask = np.zeros((1280, 1280), dtype=bool)
    mask[:160, :160] = True
    p = _write_mask(tmp_path / "a1.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    assert cov["A1"] == 1.0
    assert cov["A2"] == 0.0
    assert cov["B1"] == 0.0


def test_partial_cell_fraction(tmp_path: Path):
    """A 80x80 block at (0, 0) should fill 25% of cell A1 (160x160 cell)."""
    mask = np.zeros((1280, 1280), dtype=bool)
    mask[:80, :80] = True
    p = _write_mask(tmp_path / "quarter.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    assert cov["A1"] == pytest.approx(0.25, abs=0.01)


def test_mask_is_resized_to_image_size(tmp_path: Path):
    """Coverage should work even if mask PNG is a different size."""
    mask = np.ones((800, 800), dtype=bool)  # smaller than default 1280x1280
    p = _write_mask(tmp_path / "small.png", mask)

    cov = compute_cell_coverage(p, grid_rows=8, grid_cols=8)
    for frac in cov.values():
        assert frac == 1.0


def test_grid_labels_handle_large_rows(tmp_path: Path):
    mask = np.zeros((100, 100), dtype=bool)
    p = _write_mask(tmp_path / "x.png", mask)
    cov = compute_cell_coverage(p, grid_rows=30, grid_cols=5, image_size=(100, 100))
    assert "A1" in cov
    assert "Z5" in cov
    assert "AA1" in cov  # row 26
    assert "AD5" in cov  # row 29


# --- build_evidence_table ---


def test_build_evidence_table_multilayer(tmp_path: Path):
    water = np.ones((1280, 1280), dtype=bool)
    channel = np.zeros((1280, 1280), dtype=bool)
    channel[:160, 160:320] = True  # cell A2 fully covered
    oyster = np.zeros((1280, 1280), dtype=bool)

    paths = {
        "water": _write_mask(tmp_path / "w.png", water),
        "channel": _write_mask(tmp_path / "c.png", channel),
        "oyster": _write_mask(tmp_path / "o.png", oyster),
    }
    ev = build_evidence_table(paths, grid_rows=8, grid_cols=8)
    assert set(ev["A1"].keys()) == {"water", "channel", "oyster"}
    assert ev["A1"]["water"] == 1.0
    assert ev["A1"]["channel"] == 0.0
    assert ev["A2"]["channel"] == 1.0
    assert ev["A2"]["oyster"] == 0.0


def test_build_evidence_table_empty_mask_paths_returns_all_cells(tmp_path: Path):
    """If no masks are provided, we should still get an entry per cell."""
    ev = build_evidence_table({}, grid_rows=4, grid_cols=4)
    assert len(ev) == 16
    assert "A1" in ev
    assert "D4" in ev
    assert ev["A1"] == {}


# --- format_evidence_for_prompt ---


def test_format_empty_evidence_returns_empty_string():
    assert format_evidence_for_prompt({}) == ""


def test_format_only_water_is_summarized(tmp_path: Path):
    """If no notable layer signal, the compact summary line should appear."""
    water = np.ones((1280, 1280), dtype=bool)
    channel = np.zeros((1280, 1280), dtype=bool)
    oyster = np.zeros((1280, 1280), dtype=bool)

    paths = {
        "water": _write_mask(tmp_path / "w.png", water),
        "channel": _write_mask(tmp_path / "c.png", channel),
        "oyster": _write_mask(tmp_path / "o.png", oyster),
    }
    ev = build_evidence_table(paths, grid_rows=4, grid_cols=4)
    text = format_evidence_for_prompt(ev)

    assert "EVIDENCE FROM SURVEYED GROUND-TRUTH SOURCES" in text
    assert "no cells have notable non-water signal" in text
    assert "Layer definitions" in text


def test_format_notable_cells_are_listed(tmp_path: Path):
    water = np.ones((1280, 1280), dtype=bool)
    channel = np.zeros((1280, 1280), dtype=bool)
    channel[:160, 160:320] = True  # A2 is 100% channel

    paths = {
        "water": _write_mask(tmp_path / "w.png", water),
        "channel": _write_mask(tmp_path / "c.png", channel),
    }
    ev = build_evidence_table(paths, grid_rows=8, grid_cols=8)
    text = format_evidence_for_prompt(ev)

    # The notable cell should be listed in the detailed table.
    assert "A2" in text
    assert "100%" in text  # the channel coverage percent
    # Boring cells summary should still appear for the other 63 cells.
    assert "other" in text.lower()


def test_format_includes_layer_notes():
    """Layer definition notes should always be present in the output."""
    ev = {"A1": {"water": 0.5, "channel": 0.0, "oyster": 0.0, "seagrass": 0.0}}
    text = format_evidence_for_prompt(ev)
    assert "BOATING CHANNEL" in text
    assert "die-off" in text  # seagrass caveat
    assert "NOT aquaculture" in text  # oyster caveat


def test_format_rounds_small_fractions_to_lt_one_pct(tmp_path: Path):
    # 0.3% should render as <1% (not 0%)
    ch = np.zeros((1280, 1280), dtype=bool)
    # Set ~0.3% of pixels in cell A1 (160x160 = 25600 pixels; 0.3% = ~77)
    ch[0:10, 0:8] = True  # 80 pixels / 25600 = 0.31%
    # Make A3 notable to trigger notable-cells rendering at all
    ch[:160, 320:480] = True

    p = _write_mask(tmp_path / "c.png", ch)
    ev = build_evidence_table({"channel": p}, grid_rows=8, grid_cols=8)
    text = format_evidence_for_prompt(ev)
    # A3 should render 100%; check A3 is in table
    assert "A3" in text
