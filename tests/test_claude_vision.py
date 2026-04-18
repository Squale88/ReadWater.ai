"""Tests for Claude vision API integration."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from readwater.api.claude_vision import (
    _cell_number_to_row_col,
    _extract_json_from_response,
    analyze_grid_image,
    analyze_structure_image,
)
from readwater.api.providers.placeholder import PlaceholderProvider

MARCO = (25.94, -81.73)


# --- Cell number mapping ---


def test_cell_1_maps_to_0_0():
    assert _cell_number_to_row_col(1) == (0, 0)


def test_cell_4_maps_to_0_3():
    assert _cell_number_to_row_col(4) == (0, 3)


def test_cell_13_maps_to_3_0():
    assert _cell_number_to_row_col(13) == (3, 0)


def test_cell_16_maps_to_3_3():
    assert _cell_number_to_row_col(16) == (3, 3)


def test_cell_5_maps_to_1_0():
    assert _cell_number_to_row_col(5) == (1, 0)


# --- JSON response parsing ---


def test_extract_json_from_fenced_block():
    raw = 'Here is my analysis...\n\n```json\n{"summary": "fenced", "scores": [1,2,3]}\n```'
    result = _extract_json_from_response(raw)
    assert result["summary"] == "fenced"


def test_extract_json_chain_of_thought():
    """Full chain-of-thought response: reasoning text then JSON block at the end."""
    raw = (
        "Cell 1: Open Gulf water, no structure. Score 2.\n"
        "Cell 2: Mangrove edges visible. Score 7.\n\n"
        '```json\n{"summary": "Coastal area", "sub_scores": [{"cell_number": 1, "score": 2}]}\n```'
    )
    result = _extract_json_from_response(raw)
    assert result["summary"] == "Coastal area"
    assert result["sub_scores"][0]["score"] == 2


def test_extract_json_plain():
    raw = '{"summary": "plain", "data": true}'
    result = _extract_json_from_response(raw)
    assert result["summary"] == "plain"


def test_extract_json_invalid_raises():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        _extract_json_from_response("not json at all")


# --- Mock helpers ---


def _mock_grid_response():
    return {
        "summary": "Coastal area with mixed features",
        "sub_scores": [
            {"cell_number": i, "score": 8.0 if i <= 4 else 2.0, "reasoning": f"Cell {i}"}
            for i in range(1, 17)
        ],
        "hydrology_notes": "Tidal flow visible moving northeast",
    }


def _mock_structure_response():
    return {
        "summary": "Productive grass flat",
        "fishable_features": [
            {
                "feature_type": "grass_flat",
                "description": "Shallow grass visible",
                "relative_position": "center",
                "fishing_notes": "Work the edges on incoming tide",
            }
        ],
        "tide_interaction": "Floods from the south",
        "wind_exposure": "Protected from north winds",
        "recommended_species": ["redfish", "speckled trout"],
        "access_notes": "Approach from the channel to the west",
        "overall_rating": 7.5,
    }


def _make_mock_client(response_dict: dict):
    """Create a mock AsyncAnthropic client that returns the given dict as fenced JSON."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # Simulate chain-of-thought: reasoning text then JSON block
    text = "Analysis reasoning...\n\n```json\n" + json.dumps(response_dict) + "\n```"
    mock_response.content = [MagicMock(text=text)]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


# --- analyze_grid_image ---


async def test_grid_image_returns_16_scores(tmp_path):
    src = str(tmp_path / "test.png")
    await PlaceholderProvider(size=64).fetch(MARCO, 10, src)

    mock_client = _make_mock_client(_mock_grid_response())
    with patch("readwater.api.claude_vision._get_client", return_value=mock_client):
        result = await analyze_grid_image(src, "parent context", 10, MARCO, 54.0)

    assert len(result["sub_scores"]) == 16
    assert result["summary"] == "Coastal area with mixed features"
    assert result["hydrology_notes"] == "Tidal flow visible moving northeast"


async def test_grid_image_includes_parent_context(tmp_path):
    src = str(tmp_path / "test.png")
    await PlaceholderProvider(size=64).fetch(MARCO, 10, src)

    mock_client = _make_mock_client(_mock_grid_response())
    with patch("readwater.api.claude_vision._get_client", return_value=mock_client):
        await analyze_grid_image(src, "Mangrove coastline to the south", 12, MARCO, 13.7)

    call_args = mock_client.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    text_block = next(b for b in user_content if b["type"] == "text")
    assert "Mangrove coastline to the south" in text_block["text"]


async def test_grid_image_sends_image(tmp_path):
    src = str(tmp_path / "test.png")
    await PlaceholderProvider(size=64).fetch(MARCO, 10, src)

    mock_client = _make_mock_client(_mock_grid_response())
    with patch("readwater.api.claude_vision._get_client", return_value=mock_client):
        await analyze_grid_image(src, "", 10, MARCO, 54.0)

    call_args = mock_client.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    image_block = next(b for b in user_content if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"
    assert len(image_block["source"]["data"]) > 0


# --- analyze_structure_image ---


async def test_structure_image_returns_features(tmp_path):
    src = str(tmp_path / "test.png")
    await PlaceholderProvider(size=64).fetch(MARCO, 18, src)

    mock_client = _make_mock_client(_mock_structure_response())
    with patch("readwater.api.claude_vision._get_client", return_value=mock_client):
        result = await analyze_structure_image(src, "Shallow bay area", MARCO, 0.21)

    assert result["summary"] == "Productive grass flat"
    assert len(result["fishable_features"]) == 1
    assert result["fishable_features"][0]["feature_type"] == "grass_flat"
    assert result["overall_rating"] == 7.5


async def test_structure_image_includes_context(tmp_path):
    src = str(tmp_path / "test.png")
    await PlaceholderProvider(size=64).fetch(MARCO, 18, src)

    mock_client = _make_mock_client(_mock_structure_response())
    with patch("readwater.api.claude_vision._get_client", return_value=mock_client):
        await analyze_structure_image(src, "Near oyster bar complex", MARCO, 0.21)

    call_args = mock_client.messages.create.call_args
    user_content = call_args.kwargs["messages"][0]["content"]
    text_block = next(b for b in user_content if b["type"] == "text")
    assert "Near oyster bar complex" in text_block["text"]


# --- API key ---


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from readwater.api.claude_vision import _get_client

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _get_client()


# --- Integration test ---


@pytest.mark.integration
async def test_real_grid_analysis(tmp_path):
    """Send a real satellite image to Claude for grid analysis.

    Requires both ANTHROPIC_API_KEY and GOOGLE_MAPS_API_KEY.
    Invoke with: pytest -m integration
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        pytest.skip("GOOGLE_MAPS_API_KEY not set")

    from readwater.api.providers.google_static import GoogleStaticProvider
    from readwater.pipeline.image_processing import draw_grid_overlay

    img_path = str(tmp_path / "marco.png")
    provider = GoogleStaticProvider()
    await provider.fetch(MARCO, zoom=12, output_path=img_path)

    grid_path = draw_grid_overlay(img_path)
    result = await analyze_grid_image(grid_path, "", 12, MARCO, 13.7)

    assert "summary" in result
    assert len(result["sub_scores"]) == 16
    for sc in result["sub_scores"]:
        assert 1 <= sc["cell_number"] <= 16
        assert 0 <= sc["score"] <= 10
