"""
rag/pipeline.py
---------------
Day 24: The full RAG pipeline.

    question ─► (index news if needed) ─► hybrid retrieve (embeddings + BM25, RRF)
             ─► FlashRank rerank (top 3) ─► formatted context ─► Groq ─► grounded answer

Steps covered:
  120. ``get_context(query, ticker)`` — one async call that runs the whole
       retrieval chain and returns a clean, numbered text block.
  121. ``PROMPT_TEMPLATE`` — "Using ONLY the following news context…".
  122. Demo: ask a real question about a real ticker and check the answer is
       grounded in (and cites) the retrieved snippets.

The heavy lifting (Tavily HTTP, embedding, Chroma, reranking, Groq) is
blocking, so each stage is pushed to a worker thread with ``asyncio.to_thread``
— callers (e.g. a future FastAPI endpoint or LangGraph node) can ``await`` it
without blocking the event loop.

Usage:
    uv run python -m src.rag.pipeline
    uv run python -m src.rag.pipeline AAPL "What are the latest developments for Apple?"
    uv run python -m src.rag.pipeline NVDA "What are NVDA's recent risks?" --refresh
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from src.core.llm.client import get_llm_response
from src.rag import vectorstore
from src.rag.hybrid import hybrid_retrieve
from src.rag.reranker import rerank
from src.rag.retriever import CHUNK_COLLECTION, index_news

PROMPT_TEMPLATE = """\
Using ONLY the following news context, answer the question:

{context}

Question: {query}

Rules:
- Cite the supporting source number(s) inline, e.g. [1] or [2][3].
- If the context does not contain the answer, say so plainly — do not use outside knowledge.
"""

SYSTEM_PROMPT = (
    "You are StockSage, a careful financial research assistant. "
    "Answer strictly from the provided news context and cite sources."
)

NO_CONTEXT = "No relevant news context was found."


def _has_stored_news(ticker: str) -> bool:
    res = vectorstore.get_collection(CHUNK_COLLECTION).get(
        where={"ticker": ticker}, limit=1, include=[]
    )
    return bool(res["ids"])


def format_context(docs: list[dict[str, Any]]) -> str:
    """Render reranked docs as a numbered, citable context block."""
    if not docs:
        return NO_CONTEXT
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("metadata", {})
        header = f"[{i}] {meta.get('title', 'Untitled')}"
        if meta.get("date"):
            header += f" ({meta['date']})"
        url = f"\nURL: {meta['url']}" if meta.get("url") else ""
        body = " ".join(d["text"].split())
        blocks.append(f"{header}{url}\n{body}")
    return "\n\n".join(blocks)


async def get_context_docs(
    query: str,
    ticker: str,
    *,
    top_n: int = 3,
    candidates: int = 10,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Run the retrieval chain and return the reranked top-*n* docs (structured)."""
    ticker = ticker.upper().strip()
    if refresh or not await asyncio.to_thread(_has_stored_news, ticker):
        await asyncio.to_thread(index_news, ticker, chunked=True)

    hits = await asyncio.to_thread(hybrid_retrieve, query, ticker, candidates)
    return await asyncio.to_thread(rerank, query, hits, top_n)


async def get_context(
    query: str,
    ticker: str,
    *,
    top_n: int = 3,
    candidates: int = 10,
    refresh: bool = False,
) -> str:
    """
    Step 120: question → clean context block ready to paste into a prompt.

    Args:
        query: The user's question.
        ticker: Stock symbol whose news to search.
        top_n: Docs to keep after reranking.
        candidates: Hybrid results passed to the reranker.
        refresh: Re-fetch news from Tavily even if some is already stored.
            (If nothing is stored for the ticker yet, it is fetched automatically.)
    """
    docs = await get_context_docs(
        query, ticker, top_n=top_n, candidates=candidates, refresh=refresh
    )
    return format_context(docs)


def build_prompt(query: str, context: str) -> str:
    """Step 121: fill the grounding prompt template."""
    return PROMPT_TEMPLATE.format(context=context, query=query)


async def answer(query: str, ticker: str, *, refresh: bool = False) -> tuple[str, str]:
    """
    Full RAG: retrieve context, ask Groq, return ``(answer, context)``.

    Note: ``get_llm_response`` streams tokens to stdout as they arrive.
    """
    context = await get_context(query, ticker, refresh=refresh)
    if context == NO_CONTEXT:
        return f"No stored or retrievable news was found for {ticker.upper()}.", context
    reply = await asyncio.to_thread(
        get_llm_response, build_prompt(query, context), SYSTEM_PROMPT
    )
    return reply, context


# ── Demo (step 122) ──────────────────────────────────────────────
async def _main() -> None:
    args = [a for a in sys.argv[1:] if a != "--refresh"]
    refresh = "--refresh" in sys.argv
    ticker = args[0].upper() if args else "NVDA"
    question = args[1] if len(args) > 1 else f"What are {ticker}'s recent risks?"

    print(f"Ticker: {ticker}\nQuestion: {question}\n")
    context = await get_context(question, ticker, refresh=refresh)
    print("── Retrieved context (top 3 after rerank) ──")
    print(context)
    if context == NO_CONTEXT:
        return

    print("\n── Groq answer ──")
    reply = await asyncio.to_thread(
        get_llm_response, build_prompt(question, context), SYSTEM_PROMPT
    )

    # Light grounding check: did the model cite the sources it was given?
    cited = sorted({n for n in range(1, 4) if f"[{n}]" in reply})
    print(f"\nCitations used: {cited or 'none'}")


if __name__ == "__main__":
    asyncio.run(_main())
