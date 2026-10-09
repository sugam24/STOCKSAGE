"""
graph/state.py
---------------
Day 34 & Day 45: StockSageState schema for LangGraph multi-agent coordination.

Defines the shared state dictionary passed between all agent nodes:
  - Supervisor: routes execution and tracks conversation history
  - Data Agent: populates `stock_data`
  - News Agent: populates `news_context`
  - Risk Agent: populates `risk_metrics`
  - Analyst Agent: synthesizes `final_report`
  - Human Gate: captures `human_feedback`
  - All nodes: append failures to `error_log`
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def append_error(existing: list[str] | None, new_errors: list[str] | None) -> list[str]:
    """Reducer function ensuring node error logs append rather than overwrite."""
    base = list(existing or [])
    if new_errors:
        base.extend(new_errors)
    return base


def append_history(existing: list[dict[str, str]] | None, new_items: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Reducer function ensuring conversation turns append across multi-turn sessions."""
    base = list(existing or [])
    if new_items:
        base.extend(new_items)
    return base


class StockSageState(TypedDict, total=False):
    """
    Step 164 & Step 205: Shared state dictionary for StockSage multi-agent graph.

    Fields:
      ticker: Uppercase equity symbol under analysis (e.g. 'AAPL', 'NVDA').
      user_query: The natural-language query entered by the user.
      route: Execution route decided by Supervisor ('data_only', 'full_analysis', 'news_only').
      required_agents: List of agent names selected by the Supervisor.
      stock_data: Fundamental snapshot, prices, and technical signals (from Data Agent).
      news_context: Retrieved news articles and SEC 10-Q filing extracts (from News Agent).
      risk_metrics: Calculated volatility, Sharpe, beta, and max drawdown (from Risk Agent).
      final_report: Synthesized institutional investment note or targeted answer.
      human_feedback: Human-in-the-loop review instructions or approval status.
      conversation_history: Multi-turn chat message history within session.
      error_log: Cumulative list of errors encountered across node executions.
    """

    ticker: str
    user_query: str
    route: str
    required_agents: list[str]
    stock_data: dict[str, Any]
    news_context: list[str]
    risk_metrics: dict[str, Any]
    final_report: str
    human_feedback: str
    conversation_history: Annotated[list[dict[str, str]], append_history]
    error_log: Annotated[list[str], append_error]


def create_initial_state(
    ticker: str = "",
    user_query: str = "",
    conversation_history: list[dict[str, str]] | None = None,
) -> StockSageState:
    """
    Initialize a StockSageState instance for a target ticker or natural language query.
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    return {
        "ticker": clean_ticker,
        "user_query": user_query.strip(),
        "route": "",
        "required_agents": [],
        "stock_data": {},
        "news_context": [],
        "risk_metrics": {},
        "final_report": "",
        "human_feedback": "",
        "conversation_history": list(conversation_history or []),
        "error_log": [],
    }
