"""
tests/test_stocksage_graph.py
------------------------------
Unit tests for Day 40 sequential multi-agent LangGraph pipeline.
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from src.graph.state import create_initial_state
from src.graph.stocksage_graph import arun_stocksage, build_sequential_graph


def test_build_sequential_graph_structure():
    """Verify graph compiles with all four agent nodes registered."""
    graph = build_sequential_graph()
    assert graph is not None
    # Verify node names in the underlying graph
    node_keys = list(graph.nodes.keys())
    assert "data_agent" in node_keys
    assert "risk_agent" in node_keys
    assert "news_agent" in node_keys
    assert "analyst_agent" in node_keys


def test_sequential_graph_flow_mocked():
    """Verify that state flows sequentially through all 4 nodes."""
    fake_stock_data = {"ticker": "AAPL", "latest_close": 230.0, "price_history": [220.0, 230.0]}
    fake_risk_metrics = {"volatility_annualized": 0.20, "risk_level": "Moderate"}
    fake_news_context = ["[1] Record quarterly revenue announced."]
    fake_report = "# Report for AAPL\nSolid earnings."

    async def mock_data(state):
        return {"stock_data": fake_stock_data}

    async def mock_risk(state):
        return {"risk_metrics": fake_risk_metrics}

    async def mock_news(state):
        return {"news_context": fake_news_context}

    async def mock_analyst(state):
        return {"final_report": fake_report}

    with patch("src.graph.stocksage_graph.data_agent_node", side_effect=mock_data), \
         patch("src.graph.stocksage_graph.risk_agent_node", side_effect=mock_risk), \
         patch("src.graph.stocksage_graph.news_agent_node", side_effect=mock_news), \
         patch("src.graph.stocksage_graph.analyst_agent_node", side_effect=mock_analyst):

        result = asyncio.run(arun_stocksage("AAPL"))

        assert result["ticker"] == "AAPL"
        assert result["stock_data"] == fake_stock_data
        assert result["risk_metrics"] == fake_risk_metrics
        assert result["news_context"] == fake_news_context
        assert result["final_report"] == fake_report
