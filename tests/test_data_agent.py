"""
tests/test_data_agent.py
-------------------------
Unit and integration tests for Day 35 Data Agent node.
"""

import asyncio
from datetime import date
from unittest.mock import patch
import pytest
from src.agents.data_agent import data_agent_node
from src.ingestion.yfinance_client import CompanyFundamentals, PriceBar, PriceHistory


def test_data_agent_node_missing_ticker():
    """Verify data_agent_node handles missing ticker gracefully."""
    state = {"ticker": ""}
    result = asyncio.run(data_agent_node(state))
    assert "error_log" in result
    assert "No ticker specified" in result["error_log"][0]
    assert "stock_data" not in result


def test_data_agent_node_mocked_success():
    """Verify data_agent_node produces structured stock_data from mocked sources."""
    fake_fundamentals = CompanyFundamentals(
        ticker="MSFT",
        short_name="Microsoft Corporation",
        sector="Technology",
        industry="Software—Infrastructure",
        market_cap=3000000000000,
        trailing_pe=35.5,
        forward_pe=30.2,
        dividend_yield=0.007,
        fifty_two_week_high=450.0,
        fifty_two_week_low=350.0,
        summary="Tech giant specializing in software and cloud.",
    )

    fake_bars = [
        PriceBar(trade_date=date(2025, 1, i + 1), open=100.0 + i, high=102.0 + i, low=99.0 + i, close=101.0 + i, volume=1000000)
        for i in range(30)
    ]
    fake_history = PriceHistory(ticker="MSFT", period="6mo", bars=fake_bars)

    with patch("src.agents.data_agent.get_fundamentals", return_value=fake_fundamentals), \
         patch("src.agents.data_agent.get_price_history", return_value=fake_history):
        state = {"ticker": "MSFT"}
        result = asyncio.run(data_agent_node(state))

        assert "stock_data" in result
        data = result["stock_data"]
        assert data["ticker"] == "MSFT"
        assert data["short_name"] == "Microsoft Corporation"
        assert data["market_cap"] == 3000000000000
        assert data["latest_close"] == 130.0
        assert data["rsi"] is not None
        assert data["ma20"] is not None


def test_data_agent_node_exception_handling():
    """Verify data_agent_node catches exceptions and appends to error_log."""
    with patch("src.agents.data_agent.get_fundamentals", side_effect=RuntimeError("Yahoo Finance network error")):
        state = {"ticker": "INVALID"}
        result = asyncio.run(data_agent_node(state))

        assert "error_log" in result
        assert len(result["error_log"]) == 1
        assert "DataAgent error for INVALID: Yahoo Finance network error" in result["error_log"][0]
        assert result["stock_data"] == {}


@pytest.mark.integration
def test_data_agent_node_live():
    """Integration test: live fetch for AAPL."""
    state = {"ticker": "AAPL"}
    result = asyncio.run(data_agent_node(state))
    assert "stock_data" in result
    data = result["stock_data"]
    assert data["ticker"] == "AAPL"
    assert data["latest_close"] > 0
    assert data["rsi"] is not None
