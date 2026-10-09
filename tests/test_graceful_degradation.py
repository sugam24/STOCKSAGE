"""
tests/test_graceful_degradation.py
-----------------------------------
Unit tests for Day 48 Graceful Error Handling and Partial Degradation.
"""

import asyncio
from unittest.mock import patch
from src.agents.analyst_agent import build_analyst_prompt
from src.graph.stocksage_graph import build_stocksage_graph


def test_build_analyst_prompt_missing_data_warnings():
    """Verify that build_analyst_prompt flags unavailable data explicitly."""
    prompt = build_analyst_prompt(
        ticker="AAPL",
        stock_data={"status": "unavailable"},
        news_context=["[UNAVAILABLE] News and SEC filing retrieval unavailable."],
        risk_metrics={},
    )
    assert "DATA AVAILABILITY WARNINGS" in prompt
    assert "Fundamental and technical price data was unavailable" in prompt
    assert "Recent news and SEC regulatory filings were unavailable" in prompt
    assert "Quantitative risk metrics" in prompt


def test_graph_finishes_with_partial_data_on_node_failure():
    """Verify that if one node fails, the graph does not crash and finishes with a report."""
    async def mock_sup(state):
        return {"ticker": "AAPL", "route": "full_analysis"}

    # Simulate Data Agent network crash
    async def broken_data_node(state):
        return {
            "stock_data": {"status": "unavailable"},
            "error_log": ["DataAgent failed: Yahoo Finance 503 Service Unavailable"],
        }

    async def mock_news(state):
        return {"news_context": ["[1] News headline: Earnings beat"]}

    async def mock_risk(state):
        return {"risk_metrics": {"status": "unavailable"}}

    async def mock_analyst(state):
        # Generates report acknowledging missing data
        return {"final_report": "# Partial Report for AAPL\nNote: Technical data was unavailable."}

    with patch("src.graph.stocksage_graph.supervisor_node", side_effect=mock_sup), \
         patch("src.graph.stocksage_graph.data_agent_node", side_effect=broken_data_node), \
         patch("src.graph.stocksage_graph.news_agent_node", side_effect=mock_news), \
         patch("src.graph.stocksage_graph.risk_agent_node", side_effect=mock_risk), \
         patch("src.graph.stocksage_graph.analyst_agent_node", side_effect=mock_analyst):

        graph = build_stocksage_graph()
        initial_state = {
            "ticker": "AAPL",
            "user_query": "Analyze AAPL",
            "stock_data": {},
            "news_context": [],
            "risk_metrics": {},
            "final_report": "",
            "conversation_history": [],
            "error_log": [],
        }

        result = asyncio.run(graph.ainvoke(initial_state))

        assert result["ticker"] == "AAPL"
        assert result["final_report"] != ""
        assert "Partial Report" in result["final_report"]
        assert len(result["error_log"]) >= 1
        assert "Yahoo Finance 503" in result["error_log"][0]
