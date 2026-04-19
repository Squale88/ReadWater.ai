"""Tests for FWC oyster + seagrass ArcGIS REST fetchers (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from readwater.api.data_sources.fwc_habitats import (
    OYSTER_SERVICE_URL,
    SEAGRASS_SERVICE_URL,
    fetch_oyster_beds,
    fetch_seagrass,
    query_arcgis_feature_service,
)


def _feature(idx: int) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-81.75 + idx * 0.01, 26.0],
                [-81.74 + idx * 0.01, 26.0],
                [-81.74 + idx * 0.01, 26.01],
                [-81.75 + idx * 0.01, 26.01],
                [-81.75 + idx * 0.01, 26.0],
            ]],
        },
        "properties": {"OBJECTID": idx, "SOURCEDATE": 2014},
    }


def _make_response(features: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json = MagicMock(return_value={
        "type": "FeatureCollection",
        "features": features,
    })
    resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _make_client(responses: list[MagicMock]) -> MagicMock:
    """MagicMock httpx.Client that returns each response in order on .get()."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.get = MagicMock(side_effect=responses)
    return mock_client


# --- Single-page happy path ---


def test_fetch_oyster_beds_single_page(tmp_path: Path):
    features = [_feature(i) for i in range(5)]
    client = _make_client([_make_response(features)])

    out = tmp_path / "oyster.geojson"
    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        result = fetch_oyster_beds((-82.0, 26.0, -81.9, 26.1), out)

    assert out.exists()
    assert result.feature_count == 5
    assert result.source_url == OYSTER_SERVICE_URL
    assert result.bbox_4326 == (-82.0, 26.0, -81.9, 26.1)

    saved = json.loads(out.read_text())
    assert saved["type"] == "FeatureCollection"
    assert len(saved["features"]) == 5
    assert client.get.call_count == 1


def test_fetch_seagrass_single_page(tmp_path: Path):
    features = [_feature(i) for i in range(3)]
    client = _make_client([_make_response(features)])

    out = tmp_path / "seagrass.geojson"
    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        result = fetch_seagrass((-82.0, 26.0, -81.9, 26.1), out)

    assert result.feature_count == 3
    assert result.source_url == SEAGRASS_SERVICE_URL


# --- Pagination ---


def test_pagination_multiple_pages(tmp_path: Path):
    """When a page returns exactly page_size, fetcher should request the next page."""
    # page_size=3: first two pages return 3 features each, third returns 1.
    page1 = [_feature(i) for i in range(3)]
    page2 = [_feature(i) for i in range(3, 6)]
    page3 = [_feature(6)]
    client = _make_client([
        _make_response(page1),
        _make_response(page2),
        _make_response(page3),
    ])

    out = tmp_path / "multi.geojson"
    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        result = query_arcgis_feature_service(
            OYSTER_SERVICE_URL, (-82.0, 26.0, -81.9, 26.1), out, page_size=3,
        )

    assert result.feature_count == 7
    assert client.get.call_count == 3

    # Verify resultOffset values on each call
    calls = client.get.call_args_list
    offsets = [call.kwargs["params"]["resultOffset"] for call in calls]
    assert offsets == ["0", "3", "6"]


def test_pagination_stops_on_empty_response(tmp_path: Path):
    client = _make_client([_make_response([])])
    out = tmp_path / "empty.geojson"
    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        result = query_arcgis_feature_service(
            OYSTER_SERVICE_URL, (-82.0, 26.0, -81.9, 26.1), out, page_size=100,
        )

    assert result.feature_count == 0
    assert client.get.call_count == 1


# --- Query params ---


def test_query_uses_correct_bbox_and_srs(tmp_path: Path):
    client = _make_client([_make_response([])])
    out = tmp_path / "x.geojson"
    bbox = (-82.0, 26.0, -81.9, 26.1)
    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        query_arcgis_feature_service(
            OYSTER_SERVICE_URL, bbox, out,
        )

    params = client.get.call_args.kwargs["params"]
    assert params["geometry"] == "-82.0,26.0,-81.9,26.1"
    assert params["geometryType"] == "esriGeometryEnvelope"
    assert params["inSR"] == "4326"
    assert params["outSR"] == "4326"
    assert params["f"] == "geojson"
    assert params["outFields"] == "*"


# --- Error propagation ---


def test_http_error_propagates(tmp_path: Path):
    resp = MagicMock()
    resp.raise_for_status = MagicMock(
        side_effect=RuntimeError("500 Server Error"),
    )
    client = _make_client([resp])

    with patch(
        "readwater.api.data_sources.fwc_habitats.httpx.Client",
        return_value=client,
    ):
        with pytest.raises(RuntimeError, match="500 Server Error"):
            query_arcgis_feature_service(
                OYSTER_SERVICE_URL, (-82.0, 26.0, -81.9, 26.1),
                tmp_path / "x.geojson",
            )
