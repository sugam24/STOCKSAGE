"""
graph/state.py
---------------
Day 34: StockSageState schema for LangGraph multi-agent coordination.

Defines the shared state dictionary passed between all agent nodes:
  - Data Agent: populates `stock_data`
  - News Agent: populates `news_context`
  - Risk Agent: populates `risk_metrics`
  - Analyst Agent: synthesizes `final_report`
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


class StockSageState(TypedDict, total=False):
    """
    Step 164: Shared state dictionary for StockSage multi-agent graph.

    Fields:
      ticker: Uppercase equity symbol under analysis (e.g. 'AAPL', 'NVDA').
      stock_data: Fundamental snapshot, prices, and technical signals (from Data Agent).
      news_context: Retrieved news articles and SEC 10-Q filing extracts (from News Agent).
      risk_metrics: Calculated volatility, Sharpe, beta, and max drawdown (from Risk Agent).
      final_report: Synthesized institutional investment note (from Analyst Agent).
      error_log: Cumulative list of errors encountered across node executions.
    """

    ticker: str
    stock_data: dict[str, Any]
    news_context: list[str]
    risk_metrics: dict[str, Any]
    final_report: str
    error_log: Annotated[list[str], append_error]


def create_initial_state(ticker: str) -> StockSageState:
    """
    Initialize an empty StockSageState instance for a target ticker symbol.
    """
    clean_ticker = ticker.strip().upper()
    return {
        "ticker": clean_ticker,
        "stock_data": {},
        "news_context": [],
        "risk_metrics": {},
        "final_report": "",
        "error_log": [],
    }
