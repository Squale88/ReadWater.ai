"""Tests for the NOAA ENC fetcher (download + extract entry points).

Real S-57 parsing is tested end-to-end via the live CLI; here we unit-test
the HTTP download path (mocked) and the .000 discovery heuristic.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from readwater.api.data_sources.noaa_enc import (
    _find_enc_file,
    _parse_marker,
    download_enc,
    extract_channels,
    find_charts_covering_point,
)


def _make_fake_enc_zip(
    zip_path: Path,
    chart_id: str = "US5FL13M",
    root_dir: str | None = None,
) -> None:
    """Build a minimal zip that mirrors NOAA's layout.

    NOAA ENC zips contain <chart_id>/<chart_id>.000 plus metadata. Some
    variants use ENC_ROOT/<chart_id>/<chart_id>.000. We exercise both.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        prefix = root_dir + "/" if root_dir else ""
        # The .000 file
        zf.writestr(f"{prefix}{chart_id}/{chart_id}.000", b"fake-s57-content")
        zf.writestr(f"{prefix}{chart_id}/CATALOG.031", b"fake-catalog")


# --- _find_enc_file ---


def test_find_enc_file_plain_layout(tmp_path: Path):
    root = tmp_path / "US5FL13M"
    root.mkdir()
    target = root / "US5FL13M.000"
    target.write_bytes(b"stub")

    found = _find_enc_file(tmp_path, "US5FL13M")
    assert found == target


def test_find_enc_file_nested_layout(tmp_path: Path):
    nested = tmp_path / "ENC_ROOT" / "US5FL13M"
    nested.mkdir(parents=True)
    target = nested / "US5FL13M.000"
    target.write_bytes(b"stub")

    found = _find_enc_file(tmp_path, "US5FL13M")
    assert found == target


def test_find_enc_file_recursive_fallback(tmp_path: Path):
    buried = tmp_path / "a" / "b" / "c" / "US5FL13M"
    buried.mkdir(parents=True)
    target = buried / "US5FL13M.000"
    target.write_bytes(b"stub")

    found = _find_enc_file(tmp_path, "US5FL13M")
    assert found == target


def test_find_enc_file_missing_returns_none(tmp_path: Path):
    assert _find_enc_file(tmp_path, "US5FL13M") is None


# --- download_enc (mocked HTTP) ---


def test_download_enc_fetches_and_extracts(tmp_path: Path):
    cache = tmp_path / "cache"

    # Build a fake NAIP-style zip in memory as the HTTP response.
    staging = tmp_path / "staging"
    staging.mkdir()
    zip_on_disk = staging / "response.zip"
    _make_fake_enc_zip(zip_on_disk, "US5FL13M")
    zip_bytes = zip_on_disk.read_bytes()

    mock_resp = MagicMock()
    mock_resp.content = zip_bytes
    mock_resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(return_value=mock_resp)

    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        result = download_enc("US5FL13M", cache)

    assert Path(result.zip_path).exists()
    assert Path(result.enc_file).exists()
    assert result.chart_id == "US5FL13M"
    assert result.enc_file.endswith("US5FL13M.000")


def test_download_enc_uses_cache_on_second_call(tmp_path: Path):
    cache = tmp_path / "cache"

    # Prime the cache
    staging = tmp_path / "staging"
    staging.mkdir()
    zip_on_disk = staging / "response.zip"
    _make_fake_enc_zip(zip_on_disk, "US5FL13M")
    zip_bytes = zip_on_disk.read_bytes()

    mock_resp = MagicMock()
    mock_resp.content = zip_bytes
    mock_resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(return_value=mock_resp)

    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        download_enc("US5FL13M", cache)  # populate cache
        assert mock_client.get.call_count == 1
        download_enc("US5FL13M", cache)  # should use cache
        assert mock_client.get.call_count == 1  # no new request


def test_download_enc_force_refetches(tmp_path: Path):
    cache = tmp_path / "cache"

    staging = tmp_path / "staging"
    staging.mkdir()
    zip_on_disk = staging / "response.zip"
    _make_fake_enc_zip(zip_on_disk, "US5FL13M")
    zip_bytes = zip_on_disk.read_bytes()

    mock_resp = MagicMock()
    mock_resp.content = zip_bytes
    mock_resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(return_value=mock_resp)

    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        download_enc("US5FL13M", cache)
        assert mock_client.get.call_count == 1
        download_enc("US5FL13M", cache, force=True)
        assert mock_client.get.call_count == 2


# --- find_charts_covering_point (mocked catalog) ---


_FAKE_CATALOG_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<root>
  <Header><title>ENC Product Catalog</title></Header>
  <cell>
    <name>US4FL1JT</name>
    <lname>Gordon Pass to Gullivan Bay</lname>
    <cscale>45000</cscale>
    <cov>
      <panel>
        <vertex><lat>25.8</lat><long>-81.9</long></vertex>
        <vertex><lat>26.1</lat><long>-81.9</long></vertex>
        <vertex><lat>26.1</lat><long>-81.6</long></vertex>
        <vertex><lat>25.8</lat><long>-81.6</long></vertex>
      </panel>
    </cov>
  </cell>
  <cell>
    <name>US3GC07M</name>
    <lname>Havana to Tampa Bay</lname>
    <cscale>350000</cscale>
    <cov>
      <panel>
        <vertex><lat>23.7</lat><long>-83.75</long></vertex>
        <vertex><lat>27.6</lat><long>-83.75</long></vertex>
        <vertex><lat>27.6</lat><long>-80.4</long></vertex>
        <vertex><lat>23.7</lat><long>-80.4</long></vertex>
      </panel>
    </cov>
  </cell>
  <cell>
    <name>US1AK01M</name>
    <lname>Alaska overview (irrelevant)</lname>
    <cscale>2000000</cscale>
    <cov>
      <panel>
        <vertex><lat>55.0</lat><long>-160.0</long></vertex>
        <vertex><lat>72.0</lat><long>-160.0</long></vertex>
        <vertex><lat>72.0</lat><long>-130.0</long></vertex>
        <vertex><lat>55.0</lat><long>-130.0</long></vertex>
      </panel>
    </cov>
  </cell>
</root>
"""


def _mock_catalog_client():
    mock_resp = MagicMock()
    mock_resp.content = _FAKE_CATALOG_XML
    mock_resp.raise_for_status = MagicMock(return_value=None)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(return_value=mock_resp)
    return mock_client


def test_find_charts_covering_point_returns_containing_charts():
    """A Marco Island point should match both the 1:45k and 1:350k covering charts."""
    mock_client = _mock_catalog_client()
    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        hits = find_charts_covering_point(26.01, -81.75)

    names = [h.name for h in hits]
    assert "US4FL1JT" in names
    assert "US3GC07M" in names
    assert "US1AK01M" not in names


def test_find_charts_covering_point_sorted_by_scale():
    """Most-detailed (smallest scale number) chart should be first."""
    mock_client = _mock_catalog_client()
    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        hits = find_charts_covering_point(26.01, -81.75)

    assert hits[0].name == "US4FL1JT"
    assert hits[0].compilation_scale == 45000


def test_find_charts_covering_point_respects_scale_filter():
    mock_client = _mock_catalog_client()
    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        # Only charts with 1:N between 1 and 100000 — excludes the coarse ones.
        hits = find_charts_covering_point(26.01, -81.75, max_scale=100000)

    assert [h.name for h in hits] == ["US4FL1JT"]


def test_find_charts_covering_point_no_match():
    mock_client = _mock_catalog_client()
    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        # Middle of the Atlantic — no charts should cover it.
        hits = find_charts_covering_point(30.0, -50.0)

    assert hits == []


def test_download_enc_raises_when_000_missing(tmp_path: Path):
    """If the downloaded zip doesn't contain a .000 file, we should fail loudly."""
    cache = tmp_path / "cache"

    # Build an empty zip (no .000 file)
    staging = tmp_path / "staging"
    staging.mkdir()
    bad_zip = staging / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("not_the_enc.txt", b"hello")
    bad_bytes = bad_zip.read_bytes()

    mock_resp = MagicMock()
    mock_resp.content = bad_bytes
    mock_resp.raise_for_status = MagicMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(return_value=mock_resp)

    with patch(
        "readwater.api.data_sources.noaa_enc.httpx.Client",
        return_value=mock_client,
    ):
        with pytest.raises(RuntimeError, match="did not contain a .000 file"):
            download_enc("US5FL13M", cache)


# --- extract_channels: feature-class filtering ---
#
# These tests mock fiona so we don't need a real .000 file. Each fake layer
# is a list of (geometry_dict, properties_dict) tuples that mirrors what
# fiona.open yields for an S-57 dataset. Target behavior:
#   - FAIRWY + DRGARE always included when present
#   - SEAARE included only when OBJNAM matches a channel-name keyword
#   - DEPARE included only when polygon area and DRVAL1 are both in range
#
# Polygon sizing for the DEPARE area filter (default threshold 5 km²):
# the test helper uses the same rough ``area * 111²`` approximation as the
# production code, so a 0.01° × 0.01° square is ~1.23 km² (small / passes)
# and a 0.1° × 0.1° square is ~123 km² (large / excluded).


def _poly(xmin: float, ymin: float, xmax: float, ymax: float) -> dict:
    """Axis-aligned rectangular polygon as a GeoJSON-ish dict."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin],
        ]],
    }


def _small_poly(cx: float = -81.78, cy: float = 26.02, side_deg: float = 0.01) -> dict:
    """Small polygon (~1.2 km² at default 0.01° side) — within DEPARE area cap."""
    h = side_deg / 2
    return _poly(cx - h, cy - h, cx + h, cy + h)


def _large_poly(cx: float = -81.78, cy: float = 26.02, side_deg: float = 0.1) -> dict:
    """Large polygon (~123 km² at default 0.1° side) — exceeds DEPARE area cap."""
    h = side_deg / 2
    return _poly(cx - h, cy - h, cx + h, cy + h)


class _FakeLayer:
    """Stand-in for ``fiona.open(...)`` — iterable context manager of features."""

    def __init__(self, features: list[dict]):
        self._features = features

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._features)


def _make_feature(geometry: dict, properties: dict) -> dict:
    return {"geometry": geometry, "properties": properties}


def _patch_fiona(layer_features: dict[str, list[dict]]):
    """Patch fiona.listlayers + fiona.open to return the given per-layer features."""
    available_layers = list(layer_features.keys())

    def fake_open(_path, layer: str):
        return _FakeLayer(layer_features.get(layer, []))

    return (
        patch("fiona.listlayers", return_value=available_layers),
        patch("fiona.open", side_effect=fake_open),
    )


def _make_point_feature(lon: float, lat: float, objnam: str, catlam: int | None) -> dict:
    return {
        "geometry": {"type": "Point", "coordinates": (lon, lat)},
        "properties": {"OBJNAM": objnam, "CATLAM": catlam},
    }


# --- _parse_marker (OBJNAM → (channel, num)) ---


def test_parse_marker_simple_daybeacon():
    assert _parse_marker("Gordon Pass Channel Daybeacon 5") == (
        "Gordon Pass Channel", 5, "",
    )


def test_parse_marker_light():
    assert _parse_marker("Gordon Pass Channel Light 6") == (
        "Gordon Pass Channel", 6, "",
    )


def test_parse_marker_lighted_buoy():
    assert _parse_marker("Gordon Pass Channel Lighted Buoy 2") == (
        "Gordon Pass Channel", 2, "",
    )


def test_parse_marker_hyphenated_channel_name():
    assert _parse_marker("Big Marco Pass - Gordon Pass Daybeacon 51") == (
        "Big Marco Pass - Gordon Pass", 51, "",
    )


def test_parse_marker_suffix_letter_captured_and_uppercased():
    # Plain markers precede their 'A' siblings under (num, suffix) sort.
    assert _parse_marker("Big Marco Pass - Gordon Pass Buoy 3A") == (
        "Big Marco Pass - Gordon Pass", 3, "A",
    )
    assert _parse_marker("Foo Channel Daybeacon 27a") == (
        "Foo Channel", 27, "A",
    )


def test_parse_marker_rejects_non_marker_name():
    # Plain SEAARE name without marker tag
    assert _parse_marker("Gordon Pass") is None
    # Marker tag present but no number at the end — does not parse
    assert _parse_marker("Obstruction Daybeacon") is None
    # Letter-only sequence ("Daybeacon C") does not parse
    assert _parse_marker("Capri Pass Daybeacon C") is None
    # Empty / None
    assert _parse_marker("") is None


# --- extract_channels: polygon layers ---


def test_fairwy_and_drgare_always_included(tmp_path: Path):
    """FAIRWY + DRGARE are unconditional channel evidence."""
    fairway = _make_feature(_small_poly(), {})
    dredged = _make_feature(_small_poly(cx=-81.77), {"DRVAL1": 1.3})

    layers = {"FAIRWY": [fairway], "DRGARE": [dredged]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out, include_marker_channels=False,
        )

    assert result.counts_by_class["FAIRWY"] == 1
    assert result.counts_by_class["DRGARE"] == 1
    assert result.feature_count == 2


def test_seaare_off_by_default(tmp_path: Path):
    """SEAARE is opt-in — by default it doesn't contribute to the mask."""
    gordon_pass = _make_feature(_small_poly(), {"OBJNAM": "Gordon Pass"})

    layers = {"SEAARE": [gordon_pass]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out, include_marker_channels=False,
        )

    assert result.counts_by_class["SEAARE"] == 0


def test_seaare_opt_in_name_filter(tmp_path: Path):
    """With include_named_channels=True, only channel-named SEAAREs pass."""
    gordon_pass = _make_feature(_small_poly(), {"OBJNAM": "Gordon Pass"})
    dollar_bay = _make_feature(_small_poly(cx=-81.77), {"OBJNAM": "Dollar Bay"})
    compass_bank = _make_feature(_small_poly(cx=-81.76),
                                 {"OBJNAM": "Compass Bank"})  # must not match

    layers = {"SEAARE": [gordon_pass, dollar_bay, compass_bank]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out,
            include_marker_channels=False,
            include_named_channels=True,
        )

    assert result.counts_by_class["SEAARE"] == 1
    payload = json.loads(out.read_text())
    names = {f["properties"].get("OBJNAM") for f in payload["features"]}
    assert names == {"Gordon Pass"}


def test_depare_off_by_default(tmp_path: Path):
    """DEPARE is opt-in — bathymetric zones are excluded from the mask unless
    the caller specifically wants a depth overlay."""
    channel_body = _make_feature(_small_poly(), {"DRVAL1": 1.8, "DRVAL2": 3.6})

    layers = {"DEPARE": [channel_body]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out, include_marker_channels=False,
        )

    assert result.counts_by_class["DEPARE"] == 0


def test_depare_opt_in_area_and_depth_filter(tmp_path: Path):
    """With include_depare=True, area and DRVAL1 must both be in range."""
    in_range = _make_feature(_small_poly(cx=-81.80),
                             {"DRVAL1": 1.8, "DRVAL2": 3.6})
    too_large = _make_feature(_large_poly(cx=-81.79),
                              {"DRVAL1": 1.8, "DRVAL2": 3.6})
    too_shallow = _make_feature(_small_poly(cx=-81.78),
                                {"DRVAL1": 0.0, "DRVAL2": 0.9})
    too_deep = _make_feature(_small_poly(cx=-81.77),
                             {"DRVAL1": 9.1, "DRVAL2": 18.2})
    no_depth = _make_feature(_small_poly(cx=-81.76), {"DRVAL1": None})

    layers = {"DEPARE": [in_range, too_large, too_shallow, too_deep, no_depth]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out,
            include_marker_channels=False,
            include_depare=True,
        )

    assert result.counts_by_class["DEPARE"] == 1


# --- extract_channels: lateral-marker channel polygons ---


def test_markers_two_sided_channel_forms_polygon(tmp_path: Path):
    """Green + red markers for a single channel produce one corridor polygon."""
    # Simulates the "Gordon Pass Channel" sequence: markers alternate green
    # (CATLAM=1) and red (CATLAM=2) down the channel.
    markers = [
        _make_point_feature(-81.807, 26.092, "Gordon Pass Channel Daybeacon 1", 1),
        _make_point_feature(-81.807, 26.091, "Gordon Pass Channel Daybeacon 2", 2),
        _make_point_feature(-81.804, 26.092, "Gordon Pass Channel Daybeacon 3", 1),
        _make_point_feature(-81.804, 26.091, "Gordon Pass Channel Daybeacon 4", 2),
        _make_point_feature(-81.800, 26.093, "Gordon Pass Channel Daybeacon 5", 1),
        _make_point_feature(-81.800, 26.092, "Gordon Pass Channel Daybeacon 6", 2),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 1
    payload = json.loads(out.read_text())
    feat = payload["features"][0]
    assert feat["properties"]["feature_class"] == "MARKERS"
    assert feat["properties"]["channel_name"] == "Gordon Pass Channel"
    assert feat["properties"]["n_greens"] == 3
    assert feat["properties"]["n_reds"] == 3


def test_markers_one_sided_channel_buffered_line(tmp_path: Path):
    """A channel with only green (or only red) markers still produces one
    polygon via a buffered linestring — better than silently dropping it."""
    markers = [
        _make_point_feature(-81.81, 26.10, "Factory Bay West Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Factory Bay West Channel Daybeacon 3", 1),
        _make_point_feature(-81.81, 26.08, "Factory Bay West Channel Daybeacon 5", 1),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 1
    payload = json.loads(out.read_text())
    assert payload["features"][0]["properties"]["n_greens"] == 3
    assert payload["features"][0]["properties"]["n_reds"] == 0


def test_markers_single_point_buffered(tmp_path: Path):
    """A lone marker in a channel becomes a small buffered circle."""
    markers = [
        _make_point_feature(-81.81, 26.10, "Lonely Channel Daybeacon 1", 1),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 1


def test_markers_groups_by_channel_name(tmp_path: Path):
    """Markers from different channel names produce separate polygons."""
    markers = [
        _make_point_feature(-81.81, 26.10, "Alpha Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Alpha Channel Daybeacon 2", 2),
        _make_point_feature(-81.77, 26.02, "Beta Pass Daybeacon 1", 1),
        _make_point_feature(-81.77, 26.01, "Beta Pass Daybeacon 2", 2),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 2
    payload = json.loads(out.read_text())
    names = sorted(f["properties"]["channel_name"] for f in payload["features"])
    assert names == ["Alpha Channel", "Beta Pass"]


def test_markers_respect_bbox_clip(tmp_path: Path):
    """Marker-channel polygons outside the bbox are clipped out."""
    markers = [
        _make_point_feature(-81.81, 26.10, "Inside Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Inside Channel Daybeacon 2", 2),
        _make_point_feature(-82.50, 25.10, "Outside Channel Daybeacon 1", 1),
        _make_point_feature(-82.50, 25.09, "Outside Channel Daybeacon 2", 2),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out, bbox_4326=(-81.90, 26.00, -81.70, 26.20),
        )

    # Only "Inside Channel" survives the bbox clip; "Outside Channel" is
    # counted as clipped_out.
    assert result.counts_by_class["MARKERS"] == 1
    assert result.clipped_out >= 1


def test_markers_skip_unparseable_objnam(tmp_path: Path):
    """A marker with a non-standard OBJNAM is skipped, not crashing extraction."""
    markers = [
        # Valid sibling so there's at least one channel
        _make_point_feature(-81.81, 26.10, "Alpha Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Alpha Channel Daybeacon 2", 2),
        # Unparseable
        _make_point_feature(-81.80, 26.08, "Capri Pass Daybeacon C", 1),
        _make_point_feature(-81.80, 26.07, "Some Random Buoy Without Number", 2),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 1
    payload = json.loads(out.read_text())
    assert payload["features"][0]["properties"]["channel_name"] == "Alpha Channel"


def test_markers_opt_out(tmp_path: Path):
    """include_marker_channels=False disables marker extraction."""
    markers = [
        _make_point_feature(-81.81, 26.10, "Alpha Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Alpha Channel Daybeacon 2", 2),
    ]

    layers = {"BCNLAT": markers}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out, include_marker_channels=False,
        )

    assert result.counts_by_class["MARKERS"] == 0


def test_markers_from_both_bcnlat_and_boylat(tmp_path: Path):
    """Floating buoys (BOYLAT) and fixed beacons (BCNLAT) combine into
    the same channel when their OBJNAMs share a channel name."""
    bcnlat = [
        _make_point_feature(-81.81, 26.10, "Alpha Channel Daybeacon 1", 1),
        _make_point_feature(-81.81, 26.09, "Alpha Channel Daybeacon 3", 1),
    ]
    boylat = [
        _make_point_feature(-81.80, 26.10, "Alpha Channel Buoy 2", 2),
        _make_point_feature(-81.80, 26.09, "Alpha Channel Buoy 4", 2),
    ]

    layers = {"BCNLAT": bcnlat, "BOYLAT": boylat}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.counts_by_class["MARKERS"] == 1
    payload = json.loads(out.read_text())
    assert payload["features"][0]["properties"]["n_greens"] == 2
    assert payload["features"][0]["properties"]["n_reds"] == 2


# --- bbox clip + empty chart ---


def test_extract_channels_bbox_clip(tmp_path: Path):
    """Features that don't intersect the bbox are counted as clipped_out."""
    inside = _make_feature(_poly(-81.78, 26.02, -81.76, 26.04), {})
    outside = _make_feature(_poly(-82.10, 25.70, -82.05, 25.75), {})

    layers = {"FAIRWY": [inside, outside]}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out,
            bbox_4326=(-81.90, 25.90, -81.70, 26.10),
            include_marker_channels=False,
        )

    assert result.counts_by_class["FAIRWY"] == 1
    assert result.clipped_out == 1


def test_extract_channels_missing_layers_ok(tmp_path: Path):
    """If an ENC has no relevant layers at all, return empty rather than raise."""
    layers: dict[str, list[dict]] = {}
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels("fake.000", out)

    assert result.feature_count == 0
    payload = json.loads(out.read_text())
    assert payload["features"] == []


def test_integration_mixed_layers(tmp_path: Path):
    """End-to-end with every layer: each contributes only what it should."""
    fairway = _make_feature(_small_poly(cx=-81.80), {})
    dredged = _make_feature(_small_poly(cx=-81.79), {"OBJNAM": "CUT-J"})

    gordon_pass = _make_feature(_small_poly(cx=-81.78),
                                {"OBJNAM": "Gordon Pass"})
    dollar_bay = _make_feature(_small_poly(cx=-81.77),
                               {"OBJNAM": "Dollar Bay"})

    channel_depare = _make_feature(_small_poly(cx=-81.76),
                                   {"DRVAL1": 1.8, "DRVAL2": 3.6})
    background_depare = _make_feature(_large_poly(cx=-81.75),
                                      {"DRVAL1": 5.4, "DRVAL2": 9.1})

    markers = [
        _make_point_feature(-81.74, 26.01, "Alpha Channel Daybeacon 1", 1),
        _make_point_feature(-81.74, 26.00, "Alpha Channel Daybeacon 2", 2),
    ]

    layers = {
        "FAIRWY": [fairway],
        "DRGARE": [dredged],
        "SEAARE": [gordon_pass, dollar_bay],
        "DEPARE": [channel_depare, background_depare],
        "BCNLAT": markers,
    }
    p_list, p_open = _patch_fiona(layers)

    out = tmp_path / "out.geojson"
    with p_list, p_open:
        result = extract_channels(
            "fake.000", out,
            include_named_channels=True,
            include_depare=True,
        )

    assert result.counts_by_class == {
        "FAIRWY": 1, "DRGARE": 1, "SEAARE": 1, "DEPARE": 1,
        "MARKERS": 1,
    }
    assert result.feature_count == 5
