"""
tests/test_analyst_agent.py
----------------------------
Unit tests for Day 39 Analyst Agent synthesis node.
"""

import asyncio
from unittest.mock import patch
from src.agents.analyst_agent import (
    StockReport,
    analyst_agent_node,
    build_analyst_prompt,
    render_markdown_report,
)


def test_build_analyst_prompt():
    """Verify prompt formatting compiles all input streams."""
    prompt = build_analyst_prompt(
        ticker="AAPL",
        stock_data={"price": 180.0},
        news_context=["[1] Services revenue surged"],
        risk_metrics={"volatility_annualized": 0.22},
    )
    assert "AAPL" in prompt
    assert "180.0" in prompt
    assert "Services revenue surged" in prompt
    assert "0.22" in prompt


def test_render_markdown_report():
    """Verify markdown output contains key sections."""
    report = StockReport(
        ticker="MSFT",
        summary="Solid cloud business with Azure driving double-digit growth.",
        bull_case="Copilot integration accelerates enterprise software spend.",
        bear_case="CapEx pressure from AI infrastructure buildout.",
        risk_rating="Moderate",
    )
    risk_metrics = {
        "volatility_annualized": 0.24,
        "var_95_daily": 0.021,
        "max_drawdown": -0.14,
        "beta": 1.15,
    }
    rendered = render_markdown_report(report, risk_metrics)
    assert "# 📈 StockSage Investment Research Report: MSFT" in rendered
    assert "## Executive Summary" in rendered
    assert "Solid cloud business" in rendered
    assert "## 🐂 Bull Case & Catalysts" in rendered
    assert "## 🐻 Bear Case & Downside Risks" in rendered
    assert "MODERATE" in rendered
    assert "24.0%" in rendered
    assert "1.15" in rendered


def test_analyst_agent_node_mocked():
    """Verify analyst_agent_node execution with mocked report generator."""
    mock_report = StockReport(
        ticker="AAPL",
        summary="Mock summary",
        bull_case="Mock bull",
        bear_case="Mock bear",
        risk_rating="Low",
    )

    with patch("src.agents.analyst_agent._generate_report_sync", return_value=mock_report):
        state = {
            "ticker": "AAPL",
            "stock_data": {},
            "news_context": [],
            "risk_metrics": {"volatility_annualized": 0.18},
        }
        result = asyncio.run(analyst_agent_node(state))
        assert "final_report" in result
        assert "AAPL" in result["final_report"]
        assert "Mock summary" in result["final_report"]


def test_analyst_agent_node_missing_ticker():
    """Verify error handling on missing ticker."""
    state = {"ticker": ""}
    result = asyncio.run(analyst_agent_node(state))
    assert "error_log" in result
    assert "No ticker specified" in result["error_log"][0]
