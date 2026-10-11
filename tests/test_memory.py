"""
tests/test_memory.py
---------------------
Unit tests for Day 45 short-term conversation memory within sessions.
"""

import asyncio
from unittest.mock import patch
from src.agents.supervisor import supervisor_node
from src.graph.state import append_history, create_initial_state


def test_append_history_reducer():
    """Verify non-destructive accumulation of multi-turn chat messages."""
    h1 = [{"role": "user", "content": "What is Apple's price?"}]
    h2 = [{"role": "assistant", "content": "Apple is at $230."}]
    merged = append_history(h1, h2)
    assert len(merged) == 2
    assert merged[0]["role"] == "user"
    assert merged[1]["role"] == "assistant"


def test_supervisor_resolves_pronoun_from_history():
    """Verify supervisor resolves 'it' / follow-up queries using conversation history."""
    from src.agents.supervisor import SupervisorDecision

    state = create_initial_state(
        user_query="What about its P/E ratio and RSI?",
        conversation_history=[
            {"role": "user", "content": "Tell me about NVDA."},
            {"role": "assistant", "content": "NVIDIA is trading at $125."},
        ],
    )

    def fake_decide(user_query, current_ticker="", history=None, model=""):
        # Verify history was forwarded properly to _decide_sync
        assert history is not None
        assert any("NVDA" in m.get("content", "") for m in history)
        return SupervisorDecision(
            ticker="NVDA",
            route="data_only",
            required_agents=["data_agent"],
            reasoning="Resolved pronoun 'its' to NVDA from conversation history.",
        )

    with patch("src.agents.supervisor._decide_sync", side_effect=fake_decide):
        res = asyncio.run(supervisor_node(state))
        assert res["ticker"] == "NVDA"
        assert res["route"] == "data_only"

