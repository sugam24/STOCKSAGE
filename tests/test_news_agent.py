"""
tests/test_news_agent.py
-------------------------
Unit and integration tests for Day 36 News Agent node.
"""

import asyncio
from unittest.mock import patch
import pytest
from src.agents.news_agent import news_agent_node


def test_news_agent_missing_ticker():
    """Verify news_agent_node handles missing ticker."""
    state = {"ticker": ""}
    result = asyncio.run(news_agent_node(state))
    assert "error_log" in result
    assert "No ticker specified" in result["error_log"][0]
    assert "news_context" not in result


def test_news_agent_mocked_success():
    """Verify news_agent_node formats retrieved documents."""
    mock_docs = [
        {
            "text": "Apple announced Record Q3 revenue driven by Services growth.",
            "metadata": {"title": "Apple Q3 Earnings", "collection": "news"},
            "score": 0.89,
        },
        {
            "text": "Item 1A Risk Factors: Global supply chain risks may impact iPhone shipments.",
            "metadata": {"title": "AAPL 10-Q Item 1A", "collection": "filings"},
            "score": 0.74,
        },
    ]

    with patch("src.agents.news_agent.get_context_docs", return_value=mock_docs):
        state = {"ticker": "AAPL"}
        result = asyncio.run(news_agent_node(state))

        assert "news_context" in result
        contexts = result["news_context"]
        assert len(contexts) == 2
        assert "[1] [NEWS] Apple Q3 Earnings" in contexts[0]
        assert "[2] [FILINGS] AAPL 10-Q Item 1A" in contexts[1]


def test_news_agent_error_handling():
    """Verify news_agent_node records exceptions to error_log."""
    with patch("src.agents.news_agent.get_context_docs", side_effect=RuntimeError("Tavily API 429 quota")):
        state = {"ticker": "AAPL"}
        result = asyncio.run(news_agent_node(state))

        assert "error_log" in result
        assert "Tavily API 429 quota" in result["error_log"][0]
        assert result["news_context"] == []


@pytest.mark.integration
def test_news_agent_live():
    """Integration test: live fetch for AAPL."""
    state = {"ticker": "AAPL"}
    result = asyncio.run(news_agent_node(state))
    assert "news_context" in result
    assert len(result["news_context"]) > 0
