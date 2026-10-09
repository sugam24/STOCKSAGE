"""
agents/analyst_agent.py
------------------------
Day 39: Analyst Agent node for LangGraph.

The synthesis node that reads intelligence gathered by the other three agents
(Data Agent, News Agent, Risk Agent) and prompts Groq to synthesize a comprehensive,
structured institutional research report (StockReport).

Usage:
    uv run python -m src.agents.analyst_agent
    uv run python -m src.agents.analyst_agent NVDA
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from src.graph.state import StockSageState

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class StockReport(BaseModel):
    """
    Step 182: Structured institutional equity research report schema.
    """

    ticker: str = Field(..., description="Stock ticker symbol under evaluation.")
    summary: str = Field(..., description="Executive summary of the company, financial position, and business health.")
    bull_case: str = Field(..., description="Key growth catalysts, competitive advantages, and upside potential.")
    bear_case: str = Field(..., description="Primary downside risks, competitive pressures, valuation concerns, or headwinds.")
    risk_rating: str = Field(..., description="Overall risk rating (e.g. 'Low', 'Moderate', 'Elevated', 'High', 'Critical').")


ANALYST_SYSTEM_PROMPT = """\
You are the Chief Investment Analyst at StockSage, an institutional quantitative hedge fund.
Your task is to synthesize disparate intelligence streams (market data, technical signals, \
recent news & SEC filings, and quantitative risk metrics) into an executive-grade investment report.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "ticker": "TICKER_SYMBOL",
  "summary": "2-3 paragraph executive synthesis of current corporate health, valuation, and fundamentals.",
  "bull_case": "Bullet points or structured paragraph highlighting top upside drivers and competitive catalysts.",
  "bear_case": "Bullet points or structured paragraph highlighting top downside risks, regulatory/market headwinds.",
  "risk_rating": "One of: Low | Moderate | Elevated | High | Critical"
}
"""


def build_analyst_prompt(
    ticker: str,
    stock_data: dict[str, Any],
    news_context: list[str],
    risk_metrics: dict[str, Any],
    critique: str = "",
) -> str:
    """
    Step 182, 219 & 223: Format all upstream context into a unified prompt,
    acknowledging any missing/unavailable data streams and critique feedback.
    """
    missing_notes: list[str] = []
    if not stock_data or stock_data.get("status") == "unavailable":
        missing_notes.append("• Fundamental and technical price data was unavailable for this analysis.")
    if not news_context or any("[UNAVAILABLE]" in c for c in news_context):
        missing_notes.append("• Recent news and SEC regulatory filings were unavailable for this analysis.")
    if not risk_metrics or risk_metrics.get("status") == "unavailable":
        missing_notes.append("• Quantitative risk metrics (volatility, VaR, drawdown) were unavailable.")

    missing_notice = ""
    if missing_notes:
        missing_notice = (
            "\n### ⚠️ DATA AVAILABILITY WARNINGS:\n"
            + "\n".join(missing_notes)
            + "\nIMPORTANT: You must explicitly acknowledge these missing data sources in your executive summary.\n"
        )

    critique_notice = ""
    if critique:
        critique_notice = (
            f"\n### 📝 REVISION INSTRUCTIONS FROM CRITIC (Score < 4):\n"
            f"{critique}\n"
            f"Address every critique point directly in this improved revision.\n"
        )

    news_text = "\n\n".join(news_context) if news_context else "No recent news context available."

    return f"""\
Synthesize the following intelligence packet for {ticker}:
{missing_notice}{critique_notice}
### 1. MARKET DATA & TECHNICAL INDICATORS (Data Agent):
{json.dumps(stock_data, indent=2, default=str)}

### 2. REGULATORY FILINGS & RECENT NEWS CONTEXT (News Agent):
{news_text}

### 3. QUANTITATIVE RISK METRICS (Risk Agent):
{json.dumps(risk_metrics, indent=2, default=str)}

Synthesize this into the required JSON object containing 'ticker', 'summary', 'bull_case', 'bear_case', and 'risk_rating'.
"""


def _generate_report_sync(
    ticker: str,
    stock_data: dict[str, Any],
    news_context: list[str],
    risk_metrics: dict[str, Any],
    critique: str = "",
    model: str = DEFAULT_MODEL,
) -> StockReport:
    """Synchronous worker that calls Groq with JSON mode and validates StockReport."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to .env.")

    client = Groq(api_key=api_key)
    prompt = build_analyst_prompt(ticker, stock_data, news_context, risk_metrics, critique=critique)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw_json_str = response.choices[0].message.content or "{}"
    data = json.loads(raw_json_str)

    # Ensure ticker field is present
    if "ticker" not in data or not data["ticker"]:
        data["ticker"] = ticker

    return StockReport.model_validate(data)


def render_markdown_report(report: StockReport, risk_metrics: dict[str, Any]) -> str:
    """Format the structured StockReport model into a clean institutional markdown document."""
    vol = risk_metrics.get('volatility_annualized')
    vol_str = f"{vol:.1%}" if isinstance(vol, (int, float)) else str(vol or 'N/A')
    var = risk_metrics.get('var_95_daily')
    var_str = f"{var:.1%}" if isinstance(var, (int, float)) else str(var or 'N/A')
    dd = risk_metrics.get('max_drawdown')
    dd_str = f"{dd:.1%}" if isinstance(dd, (int, float)) else str(dd or 'N/A')
    beta = risk_metrics.get('beta')
    beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else str(beta or 'N/A')

    return f"""# 📈 StockSage Investment Research Report: {report.ticker}

## Executive Summary
{report.summary}

## 🐂 Bull Case & Catalysts
{report.bull_case}

## 🐻 Bear Case & Downside Risks
{report.bear_case}

## ⚖️ Quantitative Risk Profile: {report.risk_rating.upper()}
- **Annualized Volatility:** {vol_str}
- **Historical 1-Day VaR (95%):** {var_str}
- **Max Drawdown:** {dd_str}
- **Beta:** {beta_str}
"""


async def analyst_agent_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 183 & 218: Async LangGraph node for the Analyst Agent with graceful degradation.
    """
    ticker = state.get("ticker", "").strip().upper()
    if not ticker:
        return {
            "error_log": ["AnalystAgent: No ticker specified in state."]
        }

    stock_data = state.get("stock_data") or {}
    news_context = state.get("news_context") or []
    risk_metrics = state.get("risk_metrics") or {}
    critique = state.get("critique") or ""

    try:
        report = await asyncio.to_thread(
            _generate_report_sync,
            ticker,
            stock_data,
            news_context,
            risk_metrics,
            critique,
        )

        formatted_report = render_markdown_report(report, risk_metrics)
        return {
            "final_report": formatted_report
        }

    except Exception as exc:
        error_msg = f"AnalystAgent error for {ticker}: {exc}"
        # Graceful degradation fallback report: never crash the pipeline outright
        fallback_report = (
            f"# 📈 StockSage Partial Research Report: {ticker}\n\n"
            f"**Notice:** Automated report synthesis encountered an issue ({exc}).\n\n"
            f"### Available Data Points:\n"
            f"- Price: ${stock_data.get('latest_close', 'N/A')}\n"
            f"- News Items Retrieved: {len(news_context)}\n"
            f"- Risk Metrics Status: {risk_metrics.get('risk_level', 'N/A')}\n"
        )
        return {
            "final_report": fallback_report,
            "error_log": [error_msg],
        }


if __name__ == "__main__":
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    print("=" * 65)
    print(f"  StockSage — Day 39: Analyst Agent Standalone Test ({ticker_arg})")
    print("=" * 65)

    fake_state: StockSageState = {
        "ticker": ticker_arg,
        "stock_data": {
            "short_name": "NVIDIA Corporation",
            "latest_close": 125.50,
            "rsi": 58.4,
            "ma20": 122.10,
            "ma50": 118.30,
            "market_cap": 3100000000000,
            "trailing_pe": 64.2,
        },
        "news_context": [
            "[1] [NEWS] NVIDIA expands Blackwell AI architecture production amid hyperscaler demand.",
            "[2] [FILINGS] 10-Q Item 1A: Potential export restrictions on advanced accelerators pose geographical revenue risks.",
        ],
        "risk_metrics": {
            "volatility_annualized": 0.442,
            "var_95_daily": 0.041,
            "max_drawdown": -0.224,
            "beta": 1.72,
            "risk_level": "Elevated",
        },
        "final_report": "",
        "error_log": [],
    }

    async def _test():
        print(f"Synthesizing report for {ticker_arg} with Groq...")
        result = await analyst_agent_node(fake_state)
        report = result.get("final_report", "")
        if report:
            print("\n" + report)
        else:
            print("❌ Failed:", result.get("error_log"))

    asyncio.run(_test())
