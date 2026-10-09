"""
agents/data_agent.py
---------------------
Day 35: Data Agent node for LangGraph.

Fetches company fundamentals and historical price series via Yahoo Finance,
computes technical signals (RSI-14, MA20, MA50), and returns structured
stock data back into StockSageState.

Usage:
    uv run python -m src.agents.data_agent
"""

from __future__ import annotations

import asyncio
from typing import Any
from src.graph.state import StockSageState
from src.ingestion.technical import compute_technical_signals
from src.ingestion.yfinance_client import get_fundamentals, get_price_history


def _fetch_sync_data(ticker: str) -> dict[str, Any]:
    """Synchronous worker fetching fundamentals and calculating technicals."""
    fundamentals = get_fundamentals(ticker)
    price_history = get_price_history(ticker, period="6mo")

    closing_prices = [bar.close for bar in price_history.bars]
    technicals = compute_technical_signals(closing_prices, ticker=ticker)

    return {
        "ticker": ticker,
        "fundamentals": fundamentals.model_dump(),
        "technicals": technicals.model_dump(),
        "price_history": closing_prices,
        "latest_close": technicals.latest_close,
        "rsi": technicals.rsi,
        "ma20": technicals.ma20,
        "ma50": technicals.ma50,
        "market_cap": fundamentals.market_cap,
        "trailing_pe": fundamentals.trailing_pe,
        "forward_pe": fundamentals.forward_pe,
        "dividend_yield": fundamentals.dividend_yield,
        "fifty_two_week_high": fundamentals.fifty_two_week_high,
        "fifty_two_week_low": fundamentals.fifty_two_week_low,
        "short_name": fundamentals.short_name,
        "sector": fundamentals.sector,
        "industry": fundamentals.industry,
        "summary": fundamentals.summary,
    }


async def data_agent_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 167: Async LangGraph node for the Data Agent.

    Reads `state['ticker']`, fetches fundamentals and technicals,
    and returns a state update containing `stock_data`.
    """
    ticker = state.get("ticker", "").strip().upper()
    if not ticker:
        return {
            "error_log": ["DataAgent: No ticker specified in state."]
        }

    try:
        # Offload blocking network I/O and pandas calculations to a thread pool
        stock_data = await asyncio.to_thread(_fetch_sync_data, ticker)
        return {
            "stock_data": stock_data
        }
    except Exception as exc:
        error_msg = f"DataAgent error for {ticker}: {exc}"
        return {
            "stock_data": {},
            "error_log": [error_msg],
        }


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 35: Data Agent Standalone Test")
    print("=" * 65)

    fake_state: StockSageState = {
        "ticker": "AAPL",
        "stock_data": {},
        "news_context": [],
        "risk_metrics": {},
        "final_report": "",
        "error_log": [],
    }

    async def _test():
        print(f"Calling data_agent_node for ticker: {fake_state['ticker']}...")
        result = await data_agent_node(fake_state)
        if "stock_data" in result and result["stock_data"]:
            data = result["stock_data"]
            print("\n✅ Data Agent Output Received:")
            print(f"   • Company: {data.get('short_name')} ({data.get('ticker')})")
            print(f"   • Latest Close: ${data.get('latest_close'):.2f}")
            print(f"   • RSI-14: {data.get('rsi'):.2f}")
            print(f"   • MA20: ${data.get('ma20'):.2f} | MA50: ${data.get('ma50'):.2f}")
            print(f"   • Market Cap: ${data.get('market_cap', 0):,}")
            print(f"   • Trailing P/E: {data.get('trailing_pe')}")
        else:
            print(f"\n❌ Error update: {result.get('error_log')}")

    asyncio.run(_test())
