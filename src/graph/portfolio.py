"""
graph/portfolio.py
-------------------
Day 47: Portfolio Analysis across multiple tickers.

Runs Data and Risk agents concurrently across a list of tickers using
asyncio.gather, computes pairwise daily return correlations, and calculates
an aggregate portfolio-level risk assessment.

Usage:
    uv run python -m src.graph.portfolio AAPL MSFT NVDA
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import Any
from pydantic import BaseModel, Field
from src.agents.data_agent import data_agent_node
from src.agents.risk_agent import risk_agent_node
from src.agents.risk_metrics import compute_daily_returns
from src.graph.state import create_initial_state


class TickerRiskSummary(BaseModel):
    ticker: str
    company_name: str
    latest_close: float
    rsi: float | None = None
    volatility: float
    var_95: float
    max_drawdown: float
    beta: float | None = None
    risk_level: str


class PortfolioAnalysisResult(BaseModel):
    tickers: list[str]
    assets: list[TickerRiskSummary]
    correlation_matrix: dict[str, dict[str, float]]
    average_correlation: float
    portfolio_volatility_avg: float
    diversification_score: str
    summary_markdown: str


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two equal-length return series."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0

    x_slice = x[-n:]
    y_slice = y[-n:]

    mean_x = sum(x_slice) / n
    mean_y = sum(y_slice) / n

    cov = sum((x_slice[i] - mean_x) * (y_slice[i] - mean_y) for i in range(n))
    var_x = sum((val - mean_x) ** 2 for val in x_slice)
    var_y = sum((val - mean_y) ** 2 for val in y_slice)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0

    corr = cov / denom
    # Clamp between -1.0 and 1.0 to handle floating point precision
    return float(max(-1.0, min(1.0, corr)))


def compute_portfolio_correlations(
    returns_by_ticker: dict[str, list[float]],
) -> tuple[dict[str, dict[str, float]], float]:
    """
    Step 215: Build pairwise correlation matrix and calculate average correlation.
    """
    tickers = list(returns_by_ticker.keys())
    matrix: dict[str, dict[str, float]] = {t: {} for t in tickers}

    pairwise_corrs: list[float] = []

    for i, t1 in enumerate(tickers):
        matrix[t1][t1] = 1.0
        for j in range(i + 1, len(tickers)):
            t2 = tickers[j]
            corr = round(compute_pearson_correlation(
                returns_by_ticker[t1],
                returns_by_ticker[t2],
            ), 4)
            matrix[t1][t2] = corr
            matrix[t2][t1] = corr
            pairwise_corrs.append(corr)

    avg_corr = round(sum(pairwise_corrs) / len(pairwise_corrs), 4) if pairwise_corrs else 1.0
    return matrix, avg_corr


async def _analyze_single_ticker(ticker: str) -> dict[str, Any]:
    """Run data_agent and risk_agent in sequence for a single ticker."""
    state = create_initial_state(ticker=ticker)

    # 1. Fetch data & technicals
    data_res = await data_agent_node(state)
    state["stock_data"] = data_res.get("stock_data", {})

    # 2. Compute risk metrics
    risk_res = await risk_agent_node(state)
    state["risk_metrics"] = risk_res.get("risk_metrics", {})

    return state


async def analyze_portfolio(tickers: list[str]) -> PortfolioAnalysisResult:
    """
    Step 213 & 214: Execute parallel multi-ticker portfolio analysis.
    """
    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not clean_tickers:
        raise ValueError("Must provide at least one ticker symbol.")

    # Step 214: Run all tickers in parallel using asyncio.gather
    tasks = [_analyze_single_ticker(t) for t in clean_tickers]
    results = await asyncio.gather(*tasks)

    assets: list[TickerRiskSummary] = []
    returns_by_ticker: dict[str, list[float]] = {}

    total_vol = 0.0

    for state in results:
        t = state["ticker"]
        sd = state.get("stock_data") or {}
        rm = state.get("risk_metrics") or {}

        prices = sd.get("price_history") or []
        returns_by_ticker[t] = compute_daily_returns(prices)

        vol = rm.get("volatility_annualized", 0.0)
        total_vol += vol

        summary = TickerRiskSummary(
            ticker=t,
            company_name=sd.get("short_name", t),
            latest_close=sd.get("latest_close", 0.0),
            rsi=sd.get("rsi"),
            volatility=vol,
            var_95=rm.get("var_95_daily", 0.0),
            max_drawdown=rm.get("max_drawdown", 0.0),
            beta=rm.get("beta"),
            risk_level=rm.get("risk_level", "Moderate"),
        )
        assets.append(summary)

    # Step 215: Compute pairwise correlation matrix and average correlation
    corr_matrix, avg_corr = compute_portfolio_correlations(returns_by_ticker)
    avg_vol = round(total_vol / len(assets), 4) if assets else 0.0

    # Diversification classification
    if avg_corr < 0.35:
        div_score = "Strong Diversification (low correlation between holdings)"
    elif avg_corr < 0.65:
        div_score = "Moderate Diversification"
    else:
        div_score = "High Concentration Risk (holdings move closely together)"

    # Render Markdown table
    lines = [
        f"# 💼 StockSage Multi-Ticker Portfolio Risk Analysis",
        f"\n**Holdings:** {', '.join(clean_tickers)} | **Average Correlation:** {avg_corr:.2f} | **Avg Annualized Volatility:** {avg_vol:.2%}",
        f"**Portfolio Structure:** {div_score}\n",
        "### Asset Breakdown",
        "| Ticker | Company | Price | RSI-14 | Ann. Vol | 95% 1D VaR | Max DD | Beta | Risk |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for a in assets:
        rsi_str = f"{a.rsi:.1f}" if a.rsi is not None else "-"
        beta_str = f"{a.beta:.2f}" if a.beta is not None else "-"
        lines.append(
            f"| **{a.ticker}** | {a.company_name[:20]} | ${a.latest_close:.2f} | {rsi_str} | "
            f"{a.volatility:.1%} | {a.var_95:.1%} | {a.max_drawdown:.1%} | {beta_str} | {a.risk_level} |"
        )

    lines.append("\n### Pairwise Correlation Matrix")
    lines.append("| Ticker | " + " | ".join(clean_tickers) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * len(clean_tickers)) + " |")
    for t1 in clean_tickers:
        row = [f"**{t1}**"]
        for t2 in clean_tickers:
            row.append(f"{corr_matrix[t1].get(t2, 0.0):.2f}")
        lines.append("| " + " | ".join(row) + " |")

    markdown_summary = "\n".join(lines)

    return PortfolioAnalysisResult(
        tickers=clean_tickers,
        assets=assets,
        correlation_matrix=corr_matrix,
        average_correlation=avg_corr,
        portfolio_volatility_avg=avg_vol,
        diversification_score=div_score,
        summary_markdown=markdown_summary,
    )


if __name__ == "__main__":
    cli_tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "NVDA"]
    print("=" * 70)
    print(f"  StockSage — Day 47: Portfolio Analysis ({', '.join(cli_tickers)})")
    print("=" * 70)

    res = asyncio.run(analyze_portfolio(cli_tickers))
    print("\n" + res.summary_markdown)
