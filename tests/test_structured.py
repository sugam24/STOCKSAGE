"""
tests/test_structured.py
------------------------
Day 5: Tests for core/structured.py

Coverage:
  • _make_function_declaration() produces a correctly-shaped FunctionDeclaration.
  • _convert_property() maps JSON-schema types to uppercase Gemini types.
  • extract() returns a validated StockData instance for 3 distinct paragraphs
    (live Gemini calls; requires GEMINI_API_KEY in the environment).
  • extract() raises ValueError when Gemini returns no function call (mocked).
"""

from __future__ import annotations

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
    """_convert_property() must map JSON Schema types → Gemini uppercase types."""

    def test_string_type(self):
        assert _convert_property({"type": "string"})["type"] == "STRING"

    def test_number_type(self):
        assert _convert_property({"type": "number"})["type"] == "NUMBER"

    def test_integer_type(self):
        assert _convert_property({"type": "integer"})["type"] == "INTEGER"

    def test_boolean_type(self):
        assert _convert_property({"type": "boolean"})["type"] == "BOOLEAN"

    def test_unknown_type_falls_back_to_string(self):
        assert _convert_property({"type": "exotic"})["type"] == "STRING"

    def test_description_is_forwarded(self):
        prop = {"type": "number", "description": "Stock price in USD"}
        result = _convert_property(prop)
        assert result["description"] == "Stock price in USD"

    def test_no_description_key_absent(self):
        prop = {"type": "string"}
        result = _convert_property(prop)
        assert "description" not in result


class TestMakeFunctionDeclaration:
    """_make_function_declaration() must produce a valid FunctionDeclaration."""

    def test_name_matches_model_title(self):
        decl = _make_function_declaration(StockData)
        assert decl.name == "StockData"

    def test_parameters_has_required_fields(self):
        # parameters is a google.genai.types.Schema object — use attribute access
        decl = _make_function_declaration(StockData)
        required = decl.parameters.required
        assert "ticker" in required
        assert "price" in required
        assert "pe_ratio" in required

    def test_properties_have_correct_types(self):
        decl = _make_function_declaration(StockData)
        props = decl.parameters.properties  # dict[str, Schema]
        # Schema.type is a Type enum; .value gives the uppercase string
        assert props["ticker"].type.value == "STRING"
        assert props["price"].type.value == "NUMBER"
        assert props["pe_ratio"].type.value == "NUMBER"

    def test_parameters_type_is_object(self):
        decl = _make_function_declaration(StockData)
        assert decl.parameters.type.value == "OBJECT"


class TestExtractMocked:
    """extract() error path – mocked so no network call is made."""

    def test_raises_value_error_when_no_function_call(self):
        """If Gemini returns only text, extract() must raise ValueError."""
        # Build a minimal fake response with no function_call parts
        mock_part = MagicMock()
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Sorry, I cannot extract that."

        with patch("core.structured._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="did not return a function call"):
                extract("some text", StockData)


# ===========================================================================
# Integration tests – live Gemini API (skipped if key missing)
# ===========================================================================

# Mark the entire class as "integration" so they can be excluded via:
#   uv run pytest -m "not integration"
pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
class TestExtractLive:
    """
    Live end-to-end extraction tests against the real Gemini API.

    These tests require GEMINI_API_KEY to be set in the environment or .env.
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
