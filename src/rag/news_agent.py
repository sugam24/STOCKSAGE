"""
rag/news_agent.py
-----------------
Day 30: Phase 2 Checkpoint — News RAG End-to-End.

Steps covered:
  146. Given a ticker: fetch fresh news via Tavily, index into ChromaDB,
       and answer "What are the biggest risks for this stock right now?"
       using the full RAG pipeline (chunking -> BM25 + Embeddings -> RRF -> FlashRank -> LLM).
  147. Run end-to-end flow across 3 different tickers (e.g., NVDA, AAPL, TSLA).

Usage:
    uv run python -m src.rag.news_agent NVDA
    uv run python -m src.rag.news_agent --all
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from typing import Any

from src.core.llm.client import get_llm_response
from src.rag.pipeline import build_prompt, get_context_docs, format_context, SYSTEM_PROMPT
from src.rag.retriever import index_news

DEFAULT_TICKERS = ["NVDA", "AAPL", "TSLA"]
DEFAULT_QUESTION = "What are the biggest risks for this stock right now?"


async def run_news_rag(ticker: str, question: str = DEFAULT_QUESTION, *, refresh: bool = False) -> dict[str, Any]:
    """
    Execute the complete News RAG pipeline for a given ticker.

    1. Ingest & index fresh news via Tavily
    2. Hybrid retrieve (BM25 + Dense)
    3. FlashRank rerank top passages
    4. Synthesize grounded answer
    """
    ticker = ticker.upper().strip()
    print("=" * 70)
    print(f"  News RAG Agent | Ticker: {ticker}")
    print(f"  Question: {question}")
    print("=" * 70)

    # 1. Fetch & Index
    print(f"\n[1/3] Ingesting & indexing latest news for {ticker}...")
    try:
        count = await asyncio.to_thread(index_news, ticker, chunked=True)
        print(f"      Indexed {count} chunks into ChromaDB.")
    except Exception as exc:
        print(f"      ⚠️ Ingestion notice: {exc} (using existing ChromaDB cache if available)")

    # 2. Retrieve & Rerank
    print("\n[2/3] Performing Hybrid Retrieval & FlashRank Reranking...")
    docs = await get_context_docs(question, ticker, top_n=4, candidates=20, refresh=refresh)
    context = format_context(docs)
    print(f"      Selected top {len(docs)} high-relevance chunks:")
    for i, d in enumerate(docs, start=1):
        score_val = d.get('score', 0.0)
        print(f"      [{i}] (Score: {score_val:.4f}) {d.get('metadata', {}).get('title', '')[:65]}")

    # 3. Grounded Generation
    print("\n[3/3] Generating grounded analysis...")
    prompt = build_prompt(question, context)
    try:
        reply = await asyncio.to_thread(get_llm_response, prompt, SYSTEM_PROMPT)
    except Exception as exc:
        reply = (
            f"[Note: LLM generation skipped ({exc}). "
            f"Retrieved context successfully extracted and grounded with {len(docs)} sources.]"
        )
        print(f"\n{reply}\n")

    return {
        "ticker": ticker,
        "question": question,
        "context_docs": docs,
        "answer": reply,
    }


async def main() -> None:
    args = sys.argv[1:]
    if "--all" in args:
        tickers = DEFAULT_TICKERS
    elif args and not args[0].startswith("--"):
        tickers = [args[0].upper()]
    else:
        tickers = DEFAULT_TICKERS

    print(f"\nStarting Phase 2 Checkpoint for tickers: {', '.join(tickers)}\n")
    for t in tickers:
        await run_news_rag(t, DEFAULT_QUESTION)
        print("\n" + "#" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
