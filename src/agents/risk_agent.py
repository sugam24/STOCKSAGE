"""
agents/risk_agent.py
---------------------
Day 38: Risk Agent node for LangGraph.

Computes quantitative risk metrics (Annualized Volatility, 95% Daily VaR,
Maximum Drawdown, and Beta) from the price history in StockSageState,
evaluates risk categorization, and attaches `risk_metrics` to state.

Usage:
    uv run python -m src.agents.risk_agent
    uv run python -m src.agents.risk_agent NVDA
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from src.agents.risk_metrics import compute_all_risk_metrics
from src.graph.state import StockSageState
from src.ingestion.yfinance_client import get_fundamentals, get_price_history


def categorize_risk_level(volatility: float, max_drawdown: float) -> str:
    """
    Qualitative risk categorization based on annualized volatility and max drawdown.
    """
    abs_dd = abs(max_drawdown)
    if volatility < 0.20 and abs_dd < 0.15:
        return "Low"
    if volatility < 0.35 and abs_dd < 0.25:
        return "Moderate"
    if volatility < 0.55 and abs_dd < 0.40:
        return "Elevated"
    return "High"


async def risk_agent_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 179: Async LangGraph node for the Quantitative Risk Agent.

    Reads `state['stock_data']` (or fetches price history directly for `state['ticker']`),
    computes risk metrics, and returns `{'risk_metrics': ...}`.
    """
    ticker = state.get("ticker", "").strip().upper()
    stock_data = state.get("stock_data") or {}

    prices = stock_data.get("price_history")

    try:
        # If upstream Data Agent didn't supply price history, fetch directly
        if not prices and ticker:
            history = await asyncio.to_thread(get_price_history, ticker, period="6mo")
            prices = [bar.close for bar in history.bars]

        if not prices:
            return {
                "risk_metrics": {},
                "error_log": [f"RiskAgent: No price history available for {ticker}."],
            }

        snapshot = compute_all_risk_metrics(prices)

        # Extract beta from fundamentals if available
        beta = None
        if "fundamentals" in stock_data and isinstance(stock_data["fundamentals"], dict):
            beta = stock_data["fundamentals"].get("beta")
        elif ticker:
            try:
                fund = await asyncio.to_thread(get_fundamentals, ticker)
                beta = fund.beta
            except Exception:
                pass

        risk_level = categorize_risk_level(
            snapshot.volatility_annualized,
            snapshot.max_drawdown,
        )

        metrics = {
            "ticker": ticker,
            "volatility_annualized": snapshot.volatility_annualized,
            "var_95_daily": snapshot.var_95_daily,
            "max_drawdown": snapshot.max_drawdown,
            "sample_bars": snapshot.sample_bars,
            "beta": beta,
            "risk_level": risk_level,
        }

        return {
            "risk_metrics": metrics
        }

    except Exception as exc:
        error_msg = f"RiskAgent error for {ticker}: {exc}"
        return {
            "risk_metrics": {},
            "error_log": [error_msg],
        }


if __name__ == "__main__":
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print("=" * 65)
    print(f"  StockSage — Day 38: Risk Agent Standalone Test ({ticker_arg})")
    print("=" * 65)

    fake_state: StockSageState = {
        "ticker": ticker_arg,
        "stock_data": {
            "price_history": [150.0, 155.0, 148.0, 152.0, 160.0, 158.0, 162.0, 159.0]
        },
        "news_context": [],
        "risk_metrics": {},
        "final_report": "",
        "error_log": [],
    }

    async def _test():
        print(f"Calling risk_agent_node with in-memory prices...")
        res = await risk_agent_node(fake_state)
        rm = res.get("risk_metrics", {})
        print("\n✅ Calculated Risk Metrics in State:")
        print(f"   • Volatility (Ann.): {rm.get('volatility_annualized', 0):.2%}")
        print(f"   • 95% 1-Day VaR:    {rm.get('var_95_daily', 0):.2%}")
        print(f"   • Max Drawdown:     {rm.get('max_drawdown', 0):.2%}")
        print(f"   • Risk Level:       {rm.get('risk_level')}")

    asyncio.run(_test())
