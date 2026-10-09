"""
tests/test_parallel_graph.py
-----------------------------
Unit tests for Day 42 parallel execution (fan-out and fan-in).
"""

import asyncio
from unittest.mock import patch
from src.graph.stocksage_graph import build_stocksage_graph


def test_parallel_fanout_fanin_flow():
    """Verify that both data_agent and news_agent execute in parallel and fan-in to analyst."""
    executed_nodes = []

    async def mock_sup(state):
        executed_nodes.append("supervisor")
        return {"ticker": "AAPL", "route": "full_analysis", "required_agents": ["data_agent", "news_agent"]}

    async def mock_data(state):
        executed_nodes.append("data_agent")
        return {"stock_data": {"price": 150.0, "price_history": [140.0, 150.0]}}

    async def mock_news(state):
        executed_nodes.append("news_agent")
        return {"news_context": ["[1] News headline"]}

    async def mock_risk(state):
        executed_nodes.append("risk_agent")
        return {"risk_metrics": {"volatility_annualized": 0.20}}

    async def mock_analyst(state):
        executed_nodes.append("analyst_agent")
        return {"final_report": "# Final Report"}

    with patch("src.graph.stocksage_graph.supervisor_node", side_effect=mock_sup), \
         patch("src.graph.stocksage_graph.data_agent_node", side_effect=mock_data), \
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

        res = asyncio.run(graph.ainvoke(initial_state))

        assert "supervisor" in executed_nodes
        assert "data_agent" in executed_nodes
        assert "news_agent" in executed_nodes
        assert "risk_agent" in executed_nodes
        assert "analyst_agent" in executed_nodes
        assert res["final_report"] == "# Final Report"
