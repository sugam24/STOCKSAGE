"""
tests/test_schema.py
--------------------
Step 22: pytest test verifying that invalid data raises a ValidationError.
"""

import pytest
from pydantic import ValidationError

from core.schema import StockData


class TestStockData:
    """Unit tests for the StockData Pydantic model."""

    # ── Happy-path ──────────────────────────────────────────────────────────

    def test_valid_instance(self):
        stock = StockData(ticker="AAPL", price=189.30, pe_ratio=28.5)
        assert stock.ticker == "AAPL"
        assert stock.price == pytest.approx(189.30)
        assert stock.pe_ratio == pytest.approx(28.5)

    def test_model_dump_returns_dict(self):
        stock = StockData(ticker="AAPL", price=189.30, pe_ratio=28.5)
        result = stock.model_dump()
        assert isinstance(result, dict)
        assert result == {"ticker": "AAPL", "price": 189.30, "pe_ratio": 28.5}

    def test_model_validate_from_dict(self):
        raw = {"ticker": "MSFT", "price": 415.75, "pe_ratio": 34.2}
        stock = StockData.model_validate(raw)
        assert isinstance(stock, StockData)
        assert stock.ticker == "MSFT"

    # ── Step 22: invalid data must raise ValidationError ───────────────────

    def test_invalid_price_raises_validation_error(self):
        """Non-numeric price string must not silently pass through."""
        with pytest.raises(ValidationError):
            StockData(ticker="TSLA", price="abc", pe_ratio=70.0)

    def test_invalid_pe_ratio_raises_validation_error(self):
        with pytest.raises(ValidationError):
            StockData(ticker="TSLA", price=200.0, pe_ratio="not-a-number")

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            StockData(ticker="TSLA", price=200.0)  # pe_ratio missing
