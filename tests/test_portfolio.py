"""
tests/test_portfolio.py
-------------------------
Unit tests for Day 47 Multi-Ticker Portfolio Analysis.
"""

import asyncio
from unittest.mock import patch
import pytest
from src.graph.portfolio import (
    analyze_portfolio,
    compute_pearson_correlation,
    compute_portfolio_correlations,
)


def test_compute_pearson_correlation_math():
    """Verify Pearson correlation calculations on known patterns."""
    x = [0.01, 0.02, -0.01, 0.03, -0.02]
    # Identical series should have correlation 1.0
    assert pytest.approx(compute_pearson_correlation(x, x), rel=1e-5) == 1.0

    # Inverted series should have correlation -1.0
    y = [-val for val in x]
    assert pytest.approx(compute_pearson_correlation(x, y), rel=1e-5) == -1.0


def test_compute_portfolio_correlations_matrix():
    """Verify NxN correlation matrix symmetry and diagonal."""
    rets = {
        "A": [0.01, 0.02, -0.01, 0.03],
        "B": [0.01, 0.02, -0.01, 0.03],  # perfectly correlated with A
        "C": [-0.01, -0.02, 0.01, -0.03], # perfectly negatively correlated
    }
    matrix, avg_corr = compute_portfolio_correlations(rets)
    assert matrix["A"]["A"] == 1.0
    assert matrix["B"]["B"] == 1.0
    assert pytest.approx(matrix["A"]["B"], rel=1e-4) == 1.0
    assert pytest.approx(matrix["A"]["C"], rel=1e-4) == -1.0


def test_analyze_portfolio_mocked():
    """Verify analyze_portfolio combines multi-ticker results cleanly."""
    async def mock_single_ticker(ticker):
        return {
            "ticker": ticker,
            "stock_data": {
                "short_name": f"{ticker} Inc",
                "latest_close": 150.0,
                "rsi": 55.0,
                "price_history": [140.0, 145.0, 150.0],
            },
            "risk_metrics": {
                "volatility_annualized": 0.25,
                "var_95_daily": 0.025,
                "max_drawdown": -0.15,
                "beta": 1.2,
                "risk_level": "Moderate",
            },
        }

    with patch("src.graph.portfolio._analyze_single_ticker", side_effect=mock_single_ticker):
        res = asyncio.run(analyze_portfolio(["AAPL", "MSFT", "NVDA"]))
        assert len(res.assets) == 3
        assert "AAPL" in res.correlation_matrix
        assert "MSFT" in res.correlation_matrix
        assert "NVDA" in res.correlation_matrix
        assert "### Asset Breakdown" in res.summary_markdown
        assert "| **AAPL** |" in res.summary_markdown
