"""Tests for the NOAA ENC fetcher (download + extract entry points).

Real S-57 parsing is tested end-to-end via the live CLI; here we unit-test
the HTTP download path (mocked) and the .000 discovery heuristic.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from readwater.api.data_sources.noaa_enc import (
    _find_enc_file,
    download_enc,
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
