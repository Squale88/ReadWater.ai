"""Tests for the seed-quality validator."""

from __future__ import annotations

from readwater.pipeline.structure.seed_validator import (
    Verdict,
    default_excluded_regions,
    validate_seeds,
)


IMAGE_SIZE = (1000, 1000)


def _run(positives, negatives, mode="region", excluded=None):
    return validate_seeds(
        positive_points=positives,
        negative_points=negatives,
        image_size=IMAGE_SIZE,
        extraction_mode=mode,
        excluded_regions=excluded,
    )


# --- PASS cases ---


def test_pass_simple_region_with_one_positive():
    r = _run([(500, 500)], [], mode="region")
    assert r.verdict == Verdict.PASS


def test_pass_corridor_with_two_positives():
    r = _run([(200, 500), (800, 500)], [], mode="corridor")
    assert r.verdict == Verdict.PASS


def test_pass_with_negative_far_enough():
    # 5% separation at 1000 px = 50 px. 100 px apart is fine.
    r = _run([(500, 500)], [(600, 500)], mode="region")
    assert r.verdict == Verdict.PASS


def test_pass_edge_band_with_two_positives():
    r = _run([(100, 500), (900, 500)], [], mode="edge_band")
    assert r.verdict == Verdict.PASS


# --- Minimum seeds per mode ---


def test_corridor_one_positive_regenerates():
    r = _run([(500, 500)], [], mode="corridor")
    assert r.verdict == Verdict.REGENERATE
    assert "corridor" in r.reason


def test_edge_band_one_positive_regenerates():
    r = _run([(500, 500)], [], mode="edge_band")
    assert r.verdict == Verdict.REGENERATE


def test_region_zero_positives_regenerates():
    r = _run([], [], mode="region")
    assert r.verdict == Verdict.REGENERATE


def test_point_feature_one_positive_passes():
    r = _run([(500, 500)], [], mode="point_feature")
    assert r.verdict == Verdict.PASS


# --- Out-of-bounds ---


def test_positive_out_of_bounds_regenerates():
    r = _run([(1200, 500)], [], mode="region")
    assert r.verdict == Verdict.REGENERATE
    assert "out of image bounds" in r.reason


def test_negative_out_of_bounds_regenerates():
    r = _run([(500, 500)], [(-10, 500)], mode="region")
    assert r.verdict == Verdict.REGENERATE


# --- Pos/neg separation ---


def test_positive_and_negative_too_close_regenerates():
    # 20 px < 50 px minimum
    r = _run([(500, 500)], [(520, 500)], mode="region")
    assert r.verdict == Verdict.REGENERATE
    assert "apart" in r.reason


# --- Positive clustering ---


def test_three_tightly_clustered_positives_regenerates():
    r = _run(
        [(500, 500), (501, 500), (500, 501)],
        [],
        mode="region",
    )
    assert r.verdict == Verdict.REGENERATE
    assert "clustered" in r.reason


def test_three_well_spread_positives_pass():
    r = _run(
        [(200, 200), (800, 200), (500, 800)],
        [],
        mode="region",
    )
    assert r.verdict == Verdict.PASS


def test_two_close_positives_allowed_when_only_two():
    # Clustering rule only kicks in for >=3; two close positives are allowed.
    r = _run([(500, 500), (510, 510)], [], mode="corridor")
    assert r.verdict == Verdict.PASS


# --- Excluded regions ---


def test_seed_inside_excluded_region_regenerates():
    excluded = [(0, 900, 240, 100)]  # bottom-left legend-ish area
    r = _run([(50, 950)], [], mode="region", excluded=excluded)
    assert r.verdict == Verdict.REGENERATE
    assert "overlay region" in r.reason


def test_seed_just_outside_excluded_passes():
    excluded = [(0, 900, 240, 100)]
    r = _run([(300, 950)], [], mode="region", excluded=excluded)
    assert r.verdict == Verdict.PASS


def test_default_excluded_regions_includes_legend_corner():
    regions = default_excluded_regions((1600, 1600))
    assert regions, "expected at least one default excluded region"
    # Legend is in the bottom-left; verify the exclusion covers that area.
    found = False
    for rx, ry, rw, rh in regions:
        if rx < 50 and ry > 1200:
            found = True
    assert found
