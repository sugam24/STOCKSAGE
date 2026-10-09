"""
tests/test_risk_agent.py
-------------------------
Unit tests for Day 38 Risk Agent node.
"""

import asyncio
from unittest.mock import patch
from src.agents.risk_agent import categorize_risk_level, risk_agent_node


def test_categorize_risk_level():
    """Verify qualitative risk labeling boundaries."""
    assert categorize_risk_level(0.12, -0.08) == "Low"
    assert categorize_risk_level(0.25, -0.18) == "Moderate"
    assert categorize_risk_level(0.45, -0.30) == "Elevated"
    assert categorize_risk_level(0.70, -0.50) == "High"


def test_risk_agent_node_with_in_memory_prices():
    """Verify risk_agent_node computes metrics directly from state price_history."""
    state = {
        "ticker": "TSLA",
        "stock_data": {
            "price_history": [200.0, 210.0, 195.0, 205.0, 215.0, 208.0],
            "fundamentals": {"beta": 1.95},
        },
    }

    result = asyncio.run(risk_agent_node(state))
    assert "risk_metrics" in result
    rm = result["risk_metrics"]
    assert rm["ticker"] == "TSLA"
    assert rm["volatility_annualized"] > 0
    assert rm["max_drawdown"] < 0
    assert rm["beta"] == 1.95
    assert rm["risk_level"] in ["Low", "Moderate", "Elevated", "High"]


def test_risk_agent_node_missing_data():
    """Verify risk_agent_node error handling when no prices can be found."""
    state = {"ticker": "", "stock_data": {}}
    result = asyncio.run(risk_agent_node(state))
    assert "error_log" in result
    assert "No price history available" in result["error_log"][0]
