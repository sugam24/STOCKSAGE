"""
tests/test_supervisor.py
-------------------------
Unit tests for Day 41 Supervisor agent and conditional routing.
"""

import asyncio
from unittest.mock import patch
from src.agents.supervisor import SupervisorDecision, supervisor_node
from src.graph.stocksage_graph import route_from_supervisor


def test_supervisor_decision_schema():
    """Verify validation of supervisor routing decisions."""
    dec = SupervisorDecision(
        ticker="AAPL",
        route="data_only",
        required_agents=["data_agent"],
        reasoning="User only requested current stock price.",
    )
    assert dec.ticker == "AAPL"
    assert dec.route == "data_only"
    assert dec.required_agents == ["data_agent"]


def test_supervisor_node_mocked():
    """Verify supervisor_node extracts and populates state fields."""
    mock_dec = SupervisorDecision(
        ticker="NVDA",
        route="full_analysis",
        required_agents=["data_agent", "news_agent", "risk_agent", "analyst_agent"],
        reasoning="Full investment assessment requested.",
    )

    with patch("src.agents.supervisor._decide_sync", return_value=mock_dec):
        state = {"user_query": "Give me an in-depth report on NVDA"}
        res = asyncio.run(supervisor_node(state))
        assert res["ticker"] == "NVDA"
        assert res["route"] == "full_analysis"
        assert len(res["required_agents"]) == 4


def test_route_from_supervisor():
    """Verify conditional edges branch correctly based on state route."""
    assert route_from_supervisor({"route": "data_only"}) == ["data_agent"]
    assert route_from_supervisor({"route": "news_only"}) == ["news_agent"]
    assert route_from_supervisor({"route": "full_analysis"}) == ["data_agent", "news_agent"]
