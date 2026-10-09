"""
tests/test_critic.py
---------------------
Unit tests for Day 49 Self-Reflection Critic Node.
"""

import asyncio
from unittest.mock import patch
from src.agents.critic_node import (
    CriticReview,
    critic_node,
    route_from_critic,
)


def test_critic_review_schema():
    """Verify validation of CriticReview model."""
    review = CriticReview(
        score=5,
        is_passing=True,
        completeness_score=5,
        citation_score=5,
        actionable_feedback="Exceptional report with detailed figures.",
    )
    assert review.score == 5
    assert review.is_passing is True


def test_critic_node_mocked_weak_report():
    """Verify critic_node flags weak reports and increments revision count."""
    mock_review = CriticReview(
        score=2,
        is_passing=False,
        completeness_score=2,
        citation_score=1,
        actionable_feedback="Draft lacks quantitative data and bear case analysis.",
    )

    with patch("src.agents.critic_node._review_report_sync", return_value=mock_review):
        state = {
            "ticker": "AAPL",
            "final_report": "Vague report.",
            "critique": "",
            "revision_count": 0,
        }
        res = asyncio.run(critic_node(state))
        assert "Critic Score: 2/5 (Passing: False)" in res["critique"]
        assert res["revision_count"] == 1


def test_route_from_critic_decisions():
    """Verify conditional edges route back to analyst on failure and forward on pass/cap."""
    # Failing review with revision_count = 1 -> should loop back to analyst_agent
    failing_state = {
        "critique": "[Critic Score: 2/5 (Passing: False)] Add more data.",
        "revision_count": 1,
    }
    assert route_from_critic(failing_state) == "analyst_agent"

    # Passing review -> proceed to finalize_turn
    passing_state = {
        "critique": "[Critic Score: 5/5 (Passing: True)] Excellent draft.",
        "revision_count": 1,
    }
    assert route_from_critic(passing_state) == "finalize_turn"

    # Failing review but exceeded max revisions (>= 2) -> proceed to finalize_turn
    exhausted_state = {
        "critique": "[Critic Score: 2/5 (Passing: False)] Still weak.",
        "revision_count": 3,
    }
    assert route_from_critic(exhausted_state) == "finalize_turn"
