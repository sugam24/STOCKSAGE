"""
graph/stocksage_graph.py
-------------------------
Days 40–45: Complete Multi-Agent Graph Architecture with:
  - Day 41: Supervisor Agent with intelligent conditional routing
  - Day 42: Parallel fan-out execution of Data & News agents (fan-in to Analyst)
  - Day 43: SQLite Checkpoint persistence (thread-based state recovery)
  - Day 44: Human-in-the-Loop (HITL) approval gate via LangGraph interrupt()
  - Day 45: Short-term multi-turn conversation memory within sessions

Usage:
    uv run python -m src.graph.stocksage_graph "What is AAPL's current price?"
    uv run python -m src.graph.stocksage_graph "Provide a full research report on NVDA"
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time
from typing import Any
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from src.agents.analyst_agent import analyst_agent_node
from src.agents.data_agent import data_agent_node
from src.agents.news_agent import news_agent_node
from src.agents.risk_agent import risk_agent_node
from src.agents.supervisor import supervisor_node
from src.graph.state import StockSageState, create_initial_state

CHECKPOINT_DB_PATH = os.path.join("data", "stocksage_checkpoints.db")


# ---------------------------------------------------------------------------
# Specialized Summarizer Nodes (for targeted single-agent routes)
# ---------------------------------------------------------------------------

async def data_summary_node(state: StockSageState) -> dict[str, Any]:
    """Format a concise real-time market data response when route == 'data_only'."""
    ticker = state.get("ticker", "UNKNOWN")
    data = state.get("stock_data") or {}

    close = data.get("latest_close")
    close_str = f"${close:.2f}" if close is not None else "N/A"
    rsi = data.get("rsi")
    rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
    ma20 = data.get("ma20")
    ma20_str = f"${ma20:.2f}" if ma20 is not None else "N/A"
    ma50 = data.get("ma50")
    ma50_str = f"${ma50:.2f}" if ma50 is not None else "N/A"
    pe = data.get("trailing_pe")
    pe_str = f"{pe:.2f}" if pe is not None else "N/A"
    mcap = data.get("market_cap")
    mcap_str = f"${mcap:,}" if mcap is not None else "N/A"

    report = (
        f"# 📊 Market Snapshot: {ticker}\n\n"
        f"- **Latest Close:** {close_str}\n"
        f"- **RSI (14-period):** {rsi_str}\n"
        f"- **20-Day MA:** {ma20_str} | **50-Day MA:** {ma50_str}\n"
        f"- **Trailing P/E:** {pe_str}\n"
        f"- **Market Cap:** {mcap_str}\n"
    )
    return {"final_report": report}


async def news_summary_node(state: StockSageState) -> dict[str, Any]:
    """Format a targeted news/regulatory response when route == 'news_only'."""
    ticker = state.get("ticker", "UNKNOWN")
    contexts = state.get("news_context") or []

    body = "\n\n".join(contexts) if contexts else "No recent filings or news found."
    report = f"# 📰 Regulatory & News Intelligence: {ticker}\n\n{body}\n"
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Day 44: Human-in-the-Loop Review Gate Node
# ---------------------------------------------------------------------------

def human_review_node(state: StockSageState) -> dict[str, Any]:
    """
    Step 201–203: Pause execution to let a human inspect intermediate data.
    """
    feedback = state.get("human_feedback", "")
    if not feedback:
        user_approval = interrupt({
            "message": "Human review required before report synthesis.",
            "ticker": state.get("ticker"),
            "stock_data_ready": bool(state.get("stock_data")),
            "news_context_count": len(state.get("news_context", [])),
            "risk_metrics_ready": bool(state.get("risk_metrics")),
        })
        return {"human_feedback": str(user_approval)}
    return {"human_feedback": feedback}


# ---------------------------------------------------------------------------
# Day 45: Turn Finalization Node (updates conversation memory)
# ---------------------------------------------------------------------------

def finalize_turn_node(state: StockSageState) -> dict[str, Any]:
    """
    Step 205: Record the completed user question and assistant answer in session history.
    """
    query = state.get("user_query") or state.get("ticker") or ""
    report = state.get("final_report") or ""

    new_turn = []
    if query:
        new_turn.append({"role": "user", "content": query})
    if report:
        new_turn.append({"role": "assistant", "content": report})

    return {"conversation_history": new_turn}


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_from_supervisor(state: StockSageState) -> list[str]:
    """
    Step 191 & 194: Determine downstream branching from Supervisor.
    """
    route = state.get("route", "full_analysis")
    if route == "data_only":
        return ["data_agent"]
    if route == "news_only":
        return ["news_agent"]
    # Day 42 Fan-out: execute Data Agent and News Agent in parallel
    return ["data_agent", "news_agent"]


def route_from_data(state: StockSageState) -> str:
    """Route data agent output based on selected path."""
    if state.get("route") == "data_only":
        return "data_summary"
    return "risk_agent"


def route_from_news(state: StockSageState) -> str:
    """Route news agent output based on selected path."""
    if state.get("route") == "news_only":
        return "news_summary"
    return "analyst_agent"


def route_from_human_review(state: StockSageState) -> str:
    """Evaluate human feedback gate."""
    feedback = (state.get("human_feedback") or "").strip().lower()
    if feedback.startswith("revise"):
        # Could loop back or adjust context; for now log revision
        return "analyst_agent"
    return "analyst_agent"


# ---------------------------------------------------------------------------
# Graph Builders
# ---------------------------------------------------------------------------

def build_sequential_graph():
    """
    Day 40 baseline: simple sequential graph (data -> risk -> news -> analyst).
    """
    builder = StateGraph(StockSageState)

    builder.add_node("data_agent", data_agent_node)
    builder.add_node("risk_agent", risk_agent_node)
    builder.add_node("news_agent", news_agent_node)
    builder.add_node("analyst_agent", analyst_agent_node)

    builder.add_edge(START, "data_agent")
    builder.add_edge("data_agent", "risk_agent")
    builder.add_edge("risk_agent", "news_agent")
    builder.add_edge("news_agent", "analyst_agent")
    builder.add_edge("analyst_agent", END)

    return builder.compile()


def build_stocksage_graph(
    checkpointer: Any | None = None,
    enable_hitl: bool = False,
):
    """
    Days 41–45: Production Multi-Agent Graph with Supervisor, Parallelism,
    Persistence, and Memory.
    """
    builder = StateGraph(StockSageState)

    # Register Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_agent", data_agent_node)
    builder.add_node("news_agent", news_agent_node)
    builder.add_node("risk_agent", risk_agent_node)
    builder.add_node("data_summary", data_summary_node)
    builder.add_node("news_summary", news_summary_node)
    builder.add_node("analyst_agent", analyst_agent_node)
    builder.add_node("finalize_turn", finalize_turn_node)

    if enable_hitl:
        builder.add_node("human_review", human_review_node)

    # Connect START to Supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor Conditional Branching (Day 41 & Day 42 Fan-out)
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["data_agent", "news_agent"],
    )

    # Data Agent Conditional Branching
    builder.add_conditional_edges(
        "data_agent",
        route_from_data,
        ["data_summary", "risk_agent"],
    )

    target_analyst = "human_review" if enable_hitl else "analyst_agent"

    # Risk Agent feeds into Analyst (or Human Gate)
    builder.add_edge("risk_agent", target_analyst)

    # News Agent Conditional Branching (feeds news_summary or fans-in to Analyst)
    builder.add_conditional_edges(
        "news_agent",
        lambda s: "news_summary" if s.get("route") == "news_only" else target_analyst,
        ["news_summary", target_analyst],
    )

    if enable_hitl:
        builder.add_conditional_edges(
            "human_review",
            route_from_human_review,
            ["analyst_agent"],
        )

    # All paths finalize turn memory
    builder.add_edge("data_summary", "finalize_turn")
    builder.add_edge("news_summary", "finalize_turn")
    builder.add_edge("analyst_agent", "finalize_turn")
    builder.add_edge("finalize_turn", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# High-Level Execution APIs
# ---------------------------------------------------------------------------

def get_sqlite_checkpointer(db_path: str = CHECKPOINT_DB_PATH) -> SqliteSaver:
    """Initialize or open an SQLite persistence saver."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


async def arun_stocksage_turn(
    user_query: str,
    thread_id: str = "default_session",
    checkpointer: Any | None = None,
    enable_hitl: bool = False,
    existing_history: list[dict[str, str]] | None = None,
) -> StockSageState:
    """Execute a single conversational turn through the full multi-agent system."""
    graph = build_stocksage_graph(checkpointer=checkpointer, enable_hitl=enable_hitl)
    initial_state = create_initial_state(
        user_query=user_query,
        conversation_history=existing_history,
    )
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(initial_state, config=config)


# Alias for backward compatibility with Day 40 tests
arun_stocksage = arun_stocksage_turn


def run_stocksage(ticker_or_query: str) -> StockSageState:
    """Synchronous entry point."""
    return asyncio.run(arun_stocksage_turn(ticker_or_query))


if __name__ == "__main__":
    query_input = sys.argv[1] if len(sys.argv) > 1 else "What is Apple's current share price?"
    print("=" * 70)
    print(f"  StockSage 📈  —  Full Multi-Agent System (Days 41–45)")
    print(f"  Query: \"{query_input}\"")
    print("=" * 70)

    start_t = time.perf_counter()
    res = run_stocksage(query_input)
    elapsed = time.perf_counter() - start_t

    print(f"\nExecution Finished in {elapsed:.2f}s | Route: {res.get('route')}")
    print(f"Required Agents: {res.get('required_agents')}")
    print(f"Errors: {res.get('error_log')}")
    print("\n" + "=" * 70)
    print("  RESPONSE")
    print("=" * 70)
    print(res.get("final_report"))
