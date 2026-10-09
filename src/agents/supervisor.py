"""
agents/supervisor.py
---------------------
Day 41: Multi-Agent Supervisor node with conditional routing.

Inspects the user's query and conversation history, resolves the ticker symbol,
and uses Groq to decide which specialized agents are required:
  - "data_only": price, P/E, RSI, moving averages (skips news & risk)
  - "news_only": recent headlines, regulatory filings (skips quantitative data)
  - "full_analysis": comprehensive 4-agent research report

Usage:
    uv run python -m src.agents.supervisor
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


class SupervisorDecision(BaseModel):
    """
    Step 190: Structured decision produced by the Supervisor agent.
    """

    ticker: str = Field(..., description="Resolved ticker symbol in uppercase, e.g. 'AAPL'.")
    route: str = Field(
        ...,
        description="Routing destination: 'data_only', 'news_only', or 'full_analysis'.",
    )
    required_agents: list[str] = Field(
        ...,
        description="List of agent names to execute (e.g. ['data_agent'] or all 4).",
    )
    reasoning: str = Field(..., description="Brief explanation for routing decision.")


SUPERVISOR_SYSTEM_PROMPT = """\
You are the Supervisor Agent for StockSage, a multi-agent financial intelligence system.
Your job is to inspect the user's question, determine the stock ticker, and select the minimal \
necessary specialist agents required to answer the question accurately without wasteful computation.

Available Agents:
- "data_agent": Real-time price, valuation ratios (P/E, Market Cap), and technical indicators (RSI, MA20, MA50).
- "news_agent": Recent news headlines and SEC EDGAR 10-Q filing extracts via RAG.
- "risk_agent": Quantitative risk metrics (annualized volatility, 95% 1-day VaR, max drawdown, beta).
- "analyst_agent": Institutional synthesis report combining all three sources.

Routing Rules:
1. If the user only asks for current price, quote, P/E, RSI, or technical metrics:
   route = "data_only", required_agents = ["data_agent"]
2. If the user only asks for recent news, events, headlines, or SEC filing disclosures:
   route = "news_only", required_agents = ["news_agent"]
3. If the user asks for a comprehensive assessment, deep-dive, investment thesis, risk-reward report, or provides only a ticker symbol (e.g. "AAPL"):
   route = "full_analysis", required_agents = ["data_agent", "news_agent", "risk_agent", "analyst_agent"]

If the user uses pronouns or follow-up references (e.g. "what about its price?"), resolve the ticker from the conversation history.

Respond strictly with a JSON object:
{
  "ticker": "TICKER",
  "route": "data_only" | "news_only" | "full_analysis",
  "required_agents": ["agent_1", ...],
  "reasoning": "brief explanation"
}
"""


def _decide_sync(
    user_query: str,
    current_ticker: str = "",
    history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_MODEL,
) -> SupervisorDecision:
    """Synchronous worker that invokes Groq with JSON mode."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    history_text = ""
    if history:
        turns = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-4:]]
        history_text = "\nRecent Conversation History:\n" + "\n".join(turns)

    user_prompt = (
        f"User Query: {user_query or current_ticker}\n"
        f"Default/Current Ticker Context: {current_ticker or 'None'}\n"
        f"{history_text}\n"
        "Determine the ticker and the minimal required agents."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    data = json.loads(response.choices[0].message.content or "{}")

    # Fallback sanity defaults
    resolved_ticker = (data.get("ticker") or current_ticker or "AAPL").strip().upper()
    data["ticker"] = resolved_ticker
    if not data.get("route"):
        data["route"] = "full_analysis"
    if not data.get("required_agents"):
        data["required_agents"] = (
            ["data_agent"] if data["route"] == "data_only"
            else ["news_agent"] if data["route"] == "news_only"
            else ["data_agent", "news_agent", "risk_agent", "analyst_agent"]
        )
    if not data.get("reasoning"):
        data["reasoning"] = f"Routed to {data['route']} based on query."

    return SupervisorDecision.model_validate(data)


async def supervisor_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 190: Async LangGraph node for the Supervisor Agent.

    Evaluates user_query, decides the execution route, and updates state.
    """
    user_query = state.get("user_query") or state.get("ticker") or ""
    current_ticker = state.get("ticker", "").strip().upper()
    history = state.get("conversation_history") or []

    try:
        decision = await asyncio.to_thread(
            _decide_sync,
            user_query,
            current_ticker,
            history,
        )

        return {
            "ticker": decision.ticker,
            "route": decision.route,
            "required_agents": decision.required_agents,
        }

    except Exception as exc:
        # Fallback to full analysis on supervisor error
        return {
            "ticker": current_ticker or "AAPL",
            "route": "full_analysis",
            "required_agents": ["data_agent", "news_agent", "risk_agent", "analyst_agent"],
            "error_log": [f"Supervisor error: {exc} (defaulting to full_analysis)"],
        }


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 41: Supervisor Routing Test")
    print("=" * 65)

    test_queries = [
        "What's AAPL's current share price and RSI?",
        "What are the latest regulatory risks disclosed in Tesla's SEC filings?",
        "Provide a comprehensive investment report on NVDA.",
    ]

    async def _test():
        for q in test_queries:
            print(f"\nEvaluating Query: \"{q}\"")
            state: StockSageState = {
                "ticker": "",
                "user_query": q,
                "route": "",
                "required_agents": [],
                "stock_data": {},
                "news_context": [],
                "risk_metrics": {},
                "final_report": "",
                "human_feedback": "",
                "conversation_history": [],
                "error_log": [],
            }
            res = await supervisor_node(state)
            print(f"  ➔ Ticker:          {res.get('ticker')}")
            print(f"  ➔ Route:           {res.get('route')}")
            print(f"  ➔ Required Agents: {res.get('required_agents')}")

    asyncio.run(_test())
