"""
tests/test_hitl.py
-------------------
Unit tests for Day 44 Human-in-the-Loop approval gate via LangGraph interrupt().
"""

import asyncio
from unittest.mock import patch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from src.graph.stocksage_graph import build_stocksage_graph


def test_hitl_approval_gate_pause_and_resume():
    """Verify that graph pauses at human_review and continues upon approval."""
    async def _run_test():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            async def mock_sup(state):
                return {"ticker": "AAPL", "route": "full_analysis"}

            async def mock_data(state):
                return {"stock_data": {"price": 180.0, "price_history": [170.0, 180.0]}}

            async def mock_news(state):
                return {"news_context": ["[1] Strong earnings"]}

            async def mock_risk(state):
                return {"risk_metrics": {"volatility_annualized": 0.22}}

            async def mock_analyst(state):
                return {"final_report": "# Synthesized Report After Human Approval"}

            with patch("src.graph.stocksage_graph.supervisor_node", side_effect=mock_sup), \
                 patch("src.graph.stocksage_graph.data_agent_node", side_effect=mock_data), \
                 patch("src.graph.stocksage_graph.news_agent_node", side_effect=mock_news), \
                 patch("src.graph.stocksage_graph.risk_agent_node", side_effect=mock_risk), \
                 patch("src.graph.stocksage_graph.analyst_agent_node", side_effect=mock_analyst):

                graph = build_stocksage_graph(checkpointer=saver, enable_hitl=True)

                config = {"configurable": {"thread_id": "hitl_thread_44"}}
                initial_state = {
                    "ticker": "AAPL",
                    "user_query": "Full analysis on AAPL",
                    "stock_data": {},
                    "news_context": [],
                    "risk_metrics": {},
                    "final_report": "",
                    "human_feedback": "",
                    "conversation_history": [],
                    "error_log": [],
                }

                # Step 1: Run graph - should interrupt before analyst_agent
                paused_state = await graph.ainvoke(initial_state, config=config)
                assert "__interrupt__" in paused_state
                assert paused_state["final_report"] == ""

                # Step 2: Resume with human approval
                resumed_state = await graph.ainvoke(Command(resume="approve"), config=config)
                assert resumed_state["final_report"] == "# Synthesized Report After Human Approval"
                assert resumed_state["human_feedback"] == "approve"

    asyncio.run(_run_test())
