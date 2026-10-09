"""
agents/news_agent.py
---------------------
Day 36: News Agent node for LangGraph.

Wraps the Phase 2 RAG pipeline (Tavily news + SEC 10-Q EDGAR filings,
hybrid search, BM25, and FlashRank reranking) into a LangGraph node
that populates `news_context` in StockSageState.

Usage:
    uv run python -m src.agents.news_agent
    uv run python -m src.agents.news_agent NVDA
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from src.graph.state import StockSageState
from src.rag.pipeline import format_context, get_context_docs


async def news_agent_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 170: Async LangGraph node for the News & Filings Agent.

    Retrieves top authoritative news articles and SEC filings
    for state['ticker'] and returns {'news_context': [formatted_contexts]}.
    """
    ticker = state.get("ticker", "").strip().upper()
    if not ticker:
        return {
            "error_log": ["NewsAgent: No ticker specified in state."]
        }

    query = (
        f"What are the latest key business developments, financial results, "
        f"catalysts, and major risks for {ticker}?"
    )

    try:
        # Retrieve top 4 reranked documents across filings and news
        docs = await get_context_docs(
            query=query,
            ticker=ticker,
            top_n=4,
            candidates=20,
            refresh=False,
        )

        if not docs:
            return {
                "news_context": [f"No recent news or filings found for {ticker}."]
            }

        # Format retrieved chunks with source citations
        formatted_blocks: list[str] = []
        for i, doc in enumerate(docs, start=1):
            meta = doc.get("metadata", {})
            title = meta.get("title", f"Document {i}")
            source = meta.get("collection", "news")
            text = " ".join(doc.get("text", "").split())
            score = doc.get("score", 0.0)
            formatted_blocks.append(
                f"[{i}] [{source.upper()}] {title} (Relevance: {score:.3f}):\n{text}"
            )

        return {
            "news_context": formatted_blocks
        }

    except Exception as exc:
        error_msg = f"NewsAgent error for {ticker}: {exc}"
        return {
            "news_context": [],
            "error_log": [error_msg],
        }


if __name__ == "__main__":
    ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print("=" * 65)
    print(f"  StockSage — Day 36: News Agent Standalone Test ({ticker_arg})")
    print("=" * 65)

    fake_state: StockSageState = {
        "ticker": ticker_arg,
        "stock_data": {},
        "news_context": [],
        "risk_metrics": {},
        "final_report": "",
        "error_log": [],
    }

    async def _test():
        print(f"Calling news_agent_node for ticker: {ticker_arg}...")
        result = await news_agent_node(fake_state)
        contexts = result.get("news_context", [])
        errors = result.get("error_log", [])

        if contexts:
            print(f"\n✅ Successfully retrieved {len(contexts)} news/filing chunks:")
            for item in contexts[:2]:
                print(f"\n{item[:300]}...")
        if errors:
            print(f"\n⚠️ Encountered errors: {errors}")

    asyncio.run(_test())
