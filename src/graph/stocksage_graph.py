"""
graph/stocksage_graph.py
-------------------------
Day 40: Full sequential multi-agent LangGraph pipeline.

Wires all 4 specialist agent nodes together in sequence:
    START ─► Data Agent ─► Risk Agent ─► News Agent ─► Analyst Agent ─► END

Usage:
    uv run python -m src.graph.stocksage_graph
    uv run python -m src.graph.stocksage_graph AAPL
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any
from langgraph.graph import END, START, StateGraph
from src.agents.analyst_agent import analyst_agent_node
from src.agents.data_agent import data_agent_node
from src.agents.news_agent import news_agent_node
from src.agents.risk_agent import risk_agent_node
from src.graph.state import StockSageState, create_initial_state


def build_sequential_graph():
    """
    Step 186: Build and compile the full sequential multi-agent StateGraph.
    """
    builder = StateGraph(StockSageState)

    # Register all four specialist nodes
    builder.add_node("data_agent", data_agent_node)
    builder.add_node("risk_agent", risk_agent_node)
    builder.add_node("news_agent", news_agent_node)
    builder.add_node("analyst_agent", analyst_agent_node)

    # Connect nodes sequentially:
    # START -> data_agent -> risk_agent -> news_agent -> analyst_agent -> END
    builder.add_edge(START, "data_agent")
    builder.add_edge("data_agent", "risk_agent")
    builder.add_edge("risk_agent", "news_agent")
    builder.add_edge("news_agent", "analyst_agent")
    builder.add_edge("analyst_agent", END)

    return builder.compile()


async def arun_stocksage(ticker: str) -> StockSageState:
    """
    Execute the compiled StockSage multi-agent graph asynchronously.
    """
    graph = build_sequential_graph()
    initial_state = create_initial_state(ticker)
    return await graph.ainvoke(initial_state)


def run_stocksage(ticker: str) -> StockSageState:
    """
    Synchronous entry point: run the complete 4-agent graph end-to-end.
    """
    return asyncio.run(arun_stocksage(ticker))


if __name__ == "__main__":
    ticker_input = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print("=" * 70)
    print(f"  StockSage 📈  —  Day 40: Multi-Agent Graph ({ticker_input})")
    print("  Orchestrating: Data Agent ➔ Risk Agent ➔ News Agent ➔ Analyst Agent")
    print("=" * 70)

    start_time = time.perf_counter()
    try:
        final_state = run_stocksage(ticker_input)
        elapsed = time.perf_counter() - start_time

        print(f"\nExecution completed in {elapsed:.2f}s.")
        print(f"Errors encountered: {final_state.get('error_log', [])}")

        report = final_state.get("final_report")
        if report:
            print("\n" + "=" * 70)
            print("  FINAL SYNTHESIZED RESEARCH REPORT")
            print("=" * 70)
            print(report)
        else:
            print("❌ No final report was generated. Inspect state:")
            print(final_state)

    except Exception as exc:
        print(f"❌ Execution failed: {exc}")
