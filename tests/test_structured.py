"""
tests/test_structured.py
------------------------
Day 5: Tests for core/structured.py (Groq-backed)

Coverage:
  • _make_function_declaration() produces a correctly-shaped OpenAI/Groq tool dict.
  • _convert_property() passes through lowercase JSON Schema types.
  • extract() returns a validated StockData instance for 3 distinct paragraphs
    (live Groq calls; requires GROQ_API_KEY in the environment).
  • extract() raises ValueError when Groq returns no function call (mocked).
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from core.schema import StockData
from core.structured import (
    _convert_property,
    _make_function_declaration,
    extract,
)


# ===========================================================================
# Unit tests – no network required
# ===========================================================================


class TestConvertProperty:
    """_convert_property() must pass through standard JSON Schema types."""

    def test_string_type(self):
        assert _convert_property({"type": "string"})["type"] == "string"

    def test_number_type(self):
        assert _convert_property({"type": "number"})["type"] == "number"

    def test_integer_type(self):
        assert _convert_property({"type": "integer"})["type"] == "integer"

    def test_boolean_type(self):
        assert _convert_property({"type": "boolean"})["type"] == "boolean"

    def test_unknown_type_passed_through(self):
        assert _convert_property({"type": "exotic"})["type"] == "exotic"

    def test_description_is_forwarded(self):
        prop = {"type": "number", "description": "Stock price in USD"}
        result = _convert_property(prop)
        assert result["description"] == "Stock price in USD"

    def test_no_description_key_absent(self):
        prop = {"type": "string"}
        result = _convert_property(prop)
        assert "description" not in result


class TestMakeFunctionDeclaration:
    """_make_function_declaration() must produce a valid OpenAI/Groq tool dict."""

    def test_name_matches_model_title(self):
        decl = _make_function_declaration(StockData)
        assert decl["function"]["name"] == "StockData"

    def test_parameters_has_required_fields(self):
        decl = _make_function_declaration(StockData)
        required = decl["function"]["parameters"]["required"]
        assert "ticker" in required
        assert "price" in required
        assert "pe_ratio" in required

    def test_properties_have_correct_types(self):
        decl = _make_function_declaration(StockData)
        props = decl["function"]["parameters"]["properties"]
        assert props["ticker"]["type"] == "string"
        assert props["price"]["type"] == "number"
        assert props["pe_ratio"]["type"] == "number"

    def test_parameters_type_is_object(self):
        decl = _make_function_declaration(StockData)
        assert decl["function"]["parameters"]["type"] == "object"

    def test_top_level_type_is_function(self):
        decl = _make_function_declaration(StockData)
        assert decl["type"] == "function"


class TestExtractMocked:
    """extract() error path – mocked so no network call is made."""

    def test_raises_value_error_when_no_function_call(self):
        """If Groq returns only text, extract() must raise ValueError."""
        # Build a minimal mock response with no tool_calls
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "Sorry, I cannot extract that."

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("core.structured._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="did not return a function call"):
                extract("some text", StockData)


# ===========================================================================
# Integration tests – live Groq API (skipped if key missing)
# ===========================================================================

# Mark the entire class as "integration" so they can be excluded via:
#   uv run pytest -m "not integration"
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
class TestExtractLive:
    """
    Live end-to-end extraction tests against the real Groq API.

    These tests require GROQ_API_KEY to be set in the environment or .env.
    Skip them in CI by running:  pytest -m "not integration"
    """

    # ── Paragraph 1: straightforward ────────────────────────────────────────
    def test_extract_aapl(self):
        paragraph = (
            "Apple Inc. (AAPL) closed at $189.30 today. "
            "Analysts note that its price-to-earnings ratio sits at 28.5, "
            "reflecting strong investor confidence in its services segment."
        )
        stock = extract(paragraph, StockData)

        assert isinstance(stock, StockData)
        assert stock.ticker.upper() == "AAPL"
        assert stock.price == pytest.approx(189.30, rel=1e-2)
        assert stock.pe_ratio == pytest.approx(28.5, rel=1e-2)

    # ── Paragraph 2: narrative style ─────────────────────────────────────────
    def test_extract_msft(self):
        paragraph = (
            "Microsoft shares (MSFT) have been trading around the $415 mark. "
            "The tech giant's PE ratio has risen to approximately 34.2 following "
            "its AI-driven cloud revenue beat last quarter."
        )
        stock = extract(paragraph, StockData)

        assert isinstance(stock, StockData)
        assert stock.ticker.upper() == "MSFT"
        assert stock.price == pytest.approx(415.0, rel=5e-2)  # ~$415
        assert stock.pe_ratio == pytest.approx(34.2, rel=1e-2)

    # ── Paragraph 3: brief / telegraphic ────────────────────────────────────
    def test_extract_nvda(self):
        paragraph = (
            "NVDA: price $875.40, P/E 68.3. "
            "GPU demand continues to outpace supply amid the generative-AI buildout."
        )
        stock = extract(paragraph, StockData)

        assert isinstance(stock, StockData)
        assert stock.ticker.upper() == "NVDA"
        assert stock.price == pytest.approx(875.40, rel=1e-2)
        assert stock.pe_ratio == pytest.approx(68.3, rel=1e-2)

    # ── Return-type contract ─────────────────────────────────────────────────
    def test_returns_validated_pydantic_object(self):
        """extract() must always return a proper StockData, not a raw dict."""
        paragraph = (
            "Tesla (TSLA) is priced at $248.50 with a PE ratio of 70.0."
        )
        result = extract(paragraph, StockData)
        assert isinstance(result, StockData)
        # model_dump() must work without errors
        d = result.model_dump()
        assert set(d.keys()) == {"ticker", "price", "pe_ratio"}
