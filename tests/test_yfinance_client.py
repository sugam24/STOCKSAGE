"""
tests/test_yfinance_client.py
-----------------------------
Tests for ingestion.yfinance_client — Steps 44–45.

These are **integration** tests that hit the Yahoo Finance API.
Run with:
    uv run pytest tests/test_yfinance_client.py -v -m integration
"""

import pytest

from ingestion.yfinance_client import (
    CompanyFundamentals,
    PriceBar,
    PriceHistory,
    get_fundamentals,
    get_price_history,
)


# ---------------------------------------------------------------------------
# Unit tests – pure model validation (no network)
# ---------------------------------------------------------------------------

class TestCompanyFundamentalsModel:
    """Test the CompanyFundamentals Pydantic model."""

    def test_minimal_valid(self):
        """Only the required field (ticker) is needed."""
        f = CompanyFundamentals(ticker="AAPL")
        assert f.ticker == "AAPL"
        assert f.short_name is None
        assert f.trailing_pe is None

    def test_full_valid(self):
        """All fields populated."""
        f = CompanyFundamentals(
            ticker="AAPL",
            short_name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
            trailing_pe=28.5,
            forward_pe=25.0,
            price_to_book=45.0,
            dividend_yield=0.005,
            beta=1.2,
            fifty_two_week_high=240.0,
            fifty_two_week_low=160.0,
            currency="USD",
            exchange="NMS",
            summary="Apple Inc. designs and sells consumer electronics.",
        )
        assert f.market_cap == 3_000_000_000_000
        assert f.trailing_pe == 28.5

    def test_round_trip(self):
        """model_dump → model_validate preserves data."""
        original = CompanyFundamentals(ticker="MSFT", trailing_pe=34.2)
        rebuilt = CompanyFundamentals.model_validate(original.model_dump())
        assert rebuilt == original


class TestPriceBarModel:
    """Test the PriceBar Pydantic model."""

    def test_valid(self):
        bar = PriceBar(
            trade_date="2025-01-15",
            open_price=180.0,
            high=185.0,
            low=179.5,
            close=184.0,
            volume=50_000_000,
        )
        assert bar.close == 184.0
        assert bar.volume == 50_000_000

    def test_invalid_missing_field(self):
        with pytest.raises(Exception):
            PriceBar(trade_date="2025-01-15", open_price=180.0)  # type: ignore[call-arg]  # missing high/low/close/volume


class TestPriceHistoryModel:
    """Test the PriceHistory Pydantic model."""

    def test_empty_bars(self):
        ph = PriceHistory(ticker="AAPL", period="6mo", bars=[])
        assert ph.trading_days == 0

    def test_with_bars(self):
        bars = [
            PriceBar(trade_date="2025-01-15", open_price=180.0, high=185.0, low=179.5, close=184.0, volume=50_000_000),
            PriceBar(trade_date="2025-01-16", open_price=184.0, high=186.0, low=183.0, close=185.5, volume=45_000_000),
        ]
        ph = PriceHistory(ticker="AAPL", period="1mo", bars=bars)
        assert ph.trading_days == 2
        assert ph.bars[0].trade_date.isoformat() == "2025-01-15"


# ---------------------------------------------------------------------------
# Integration tests – live API calls
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetFundamentals:
    """Integration tests for get_fundamentals()."""

    def test_aapl_returns_model(self):
        result = get_fundamentals("AAPL")
        assert isinstance(result, CompanyFundamentals)
        assert result.ticker == "AAPL"
        assert result.short_name is not None
        assert result.market_cap is not None
        assert result.market_cap > 0

    def test_case_insensitive(self):
        result = get_fundamentals("aapl")
        assert result.ticker == "AAPL"

    def test_invalid_ticker_raises(self):
        with pytest.raises(ValueError, match="No fundamental data"):
            get_fundamentals("ZZZZZZ99999")

    def test_serialization_round_trip(self):
        result = get_fundamentals("MSFT")
        raw = result.model_dump()
        rebuilt = CompanyFundamentals.model_validate(raw)
        assert rebuilt == result


@pytest.mark.integration
class TestGetPriceHistory:
    """Integration tests for get_price_history()."""

    def test_aapl_6mo(self):
        result = get_price_history("AAPL", period="6mo")
        assert isinstance(result, PriceHistory)
        assert result.ticker == "AAPL"
        assert result.period == "6mo"
        assert result.trading_days > 100  # ~125 trading days in 6 months

    def test_bars_have_correct_types(self):
        result = get_price_history("AAPL", period="1mo")
        bar = result.bars[0]
        assert isinstance(bar, PriceBar)
        assert isinstance(bar.open_price, float)
        assert isinstance(bar.volume, int)
        assert bar.high >= bar.low

    def test_default_period(self):
        """Default period parameter is '6mo'."""
        result = get_price_history("AAPL")
        assert result.period == "6mo"

    def test_invalid_ticker_raises(self):
        with pytest.raises(ValueError, match="No price history"):
            get_price_history("ZZZZZZ99999")

    def test_serialization_round_trip(self):
        result = get_price_history("AAPL", period="1mo")
        bar = result.bars[0]
        raw = bar.model_dump()
        rebuilt = PriceBar.model_validate(raw)
        assert rebuilt == bar
