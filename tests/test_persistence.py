"""
tests/test_persistence.py
--------------------------
Unit tests for Day 43 SQLite checkpoint persistence.
"""

import asyncio
from unittest.mock import patch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph.stocksage_graph import build_stocksage_graph


def test_sqlite_persistence_checkpointing():
    """Verify that graph execution state is persisted and retrievable by thread_id."""
    async def _run_test():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            async def mock_sup(state):
                return {"ticker": "MSFT", "route": "data_only"}

            async def mock_data(state):
                return {"stock_data": {"latest_close": 420.0}}

            with patch("src.graph.stocksage_graph.supervisor_node", side_effect=mock_sup), \
                 patch("src.graph.stocksage_graph.data_agent_node", side_effect=mock_data):

                graph = build_stocksage_graph(checkpointer=saver)

                thread_id = "test_session_43"
                config = {"configurable": {"thread_id": thread_id}}

                initial_state = {
                    "ticker": "MSFT",
                    "user_query": "What's MSFT price?",
                    "stock_data": {},
                    "news_context": [],
                    "risk_metrics": {},
                    "final_report": "",
                    "conversation_history": [],
                    "error_log": [],
                }

                # First execution run
                result = await graph.ainvoke(initial_state, config=config)
                assert result["ticker"] == "MSFT"
                assert result["stock_data"]["latest_close"] == 420.0

                # Retrieve saved checkpoint directly from checkpointer
                checkpoint_tuple = await saver.aget_tuple(config)
                assert checkpoint_tuple is not None
                saved_state = checkpoint_tuple.checkpoint["channel_values"]
                assert saved_state["ticker"] == "MSFT"
                assert saved_state["stock_data"]["latest_close"] == 420.0

    asyncio.run(_run_test())
