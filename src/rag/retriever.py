"""
rag/retriever.py
----------------
Day 19: Your first real retriever — Tavily → embeddings → ChromaDB → search.
Day 20: Re-index the same news as overlapping chunks instead of whole articles.

Steps covered:
   96. Fetch news for a ticker with the Day 11 Tavily client.
   97. Embed each snippet and store it in ChromaDB with metadata
       (ticker, url, date, title).
   98. ``retrieve(query, ticker)`` searches ONLY that ticker's stored news.
   99. Demo: store NVDA news and ask "What are NVDA's recent risks?".
  103. ``index_news(..., chunked=True)`` re-indexes full article text as
       ~300-word chunks (``src.rag.chunker``) into a separate collection.
  104. Demo compares whole-article vs. chunked retrieval on the same query.

Collections:
    news         → one document per article (title + Tavily snippet)   [Day 19]
    news_chunks  → full article text split into overlapping chunks     [Day 20]

Usage:
    uv run python -m src.rag.retriever                  # NVDA, default query
    uv run python -m src.rag.retriever TSLA "What is driving Tesla's stock?"
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from src.ingestion.tavily_client import filter_recent, search_news
from src.rag import vectorstore
from src.rag.chunker import chunk_text

WHOLE_COLLECTION = "news"  # Day 19
CHUNK_COLLECTION = "news_chunks"  # Day 20 — default from here on
MAX_WORDS_PER_ARTICLE = 3000  # cap very long scraped pages (nav junk, comments…)


def _doc_id(url: str, idx: int) -> str:
    """Deterministic id so re-indexing the same article upserts instead of duplicating."""
    return f"{hashlib.sha1(url.encode()).hexdigest()[:16]}-{idx}"


def fetch_news(ticker: str, *, max_results: int = 10) -> list[dict]:
    """Step 96: fetch recent news (with full article text) for *ticker*."""
    results = search_news(ticker, max_results=max_results, include_raw_content=True)
    # Prefer the last 30 days, but don't end up with nothing if dates are missing.
    return filter_recent(results) or results


def index_articles(
    articles: list[dict],
    ticker: str,
    *,
    chunked: bool = True,
) -> int:
    """
    Step 97 / 103: embed *articles* and upsert them into ChromaDB.

    Args:
        articles: Tavily result dicts (title, url, content, raw_content, published_date).
        ticker: Ticker these articles belong to (stored as metadata for filtering).
        chunked: False → one doc per article from the short snippet (Day 19).
                 True  → full article text split into overlapping chunks (Day 20).

    Returns:
        Number of documents written.
    """
    ticker = ticker.upper().strip()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []

    for art in articles:
        url = art.get("url") or ""
        title = (art.get("title") or "").strip()
        base_meta = {
            "ticker": ticker,
            "url": url,
            "title": title,
            "date": art.get("published_date") or "",
            "source": "tavily",
        }

        if not chunked:
            text = f"{title}\n{(art.get('content') or '').strip()}".strip()
            if text:
                ids.append(_doc_id(url, 0))
                docs.append(text)
                metas.append({**base_meta, "chunk_index": 0})
            continue

        body = (art.get("raw_content") or art.get("content") or "").strip()
        body = " ".join(body.split()[:MAX_WORDS_PER_ARTICLE])
        for i, chunk in enumerate(chunk_text(body)):
            ids.append(_doc_id(url, i))
            # Prefix the title so every chunk keeps its article context.
            docs.append(f"{title}\n{chunk}" if title else chunk)
            metas.append({**base_meta, "chunk_index": i})

    collection = CHUNK_COLLECTION if chunked else WHOLE_COLLECTION
    return vectorstore.add(ids, docs, metas, collection=collection)


def index_news(ticker: str, *, chunked: bool = True, max_results: int = 10) -> int:
    """Convenience: fetch + index in one call. Returns number of docs written."""
    return index_articles(fetch_news(ticker, max_results=max_results), ticker, chunked=chunked)


def retrieve(
    query: str,
    ticker: str,
    k: int = 5,
    *,
    collection: str = CHUNK_COLLECTION,
) -> list[dict[str, Any]]:
    """
    Step 98: semantic search over ONE ticker's stored news.

    Returns:
        ``[{"id", "text", "metadata", "score"}, ...]`` best match first.
    """
    return vectorstore.query(
        query,
        n_results=k,
        where={"ticker": ticker.upper().strip()},
        collection=collection,
    )


def print_hits(hits: list[dict[str, Any]], *, width: int = 220) -> None:
    """Pretty-print retrieval results."""
    if not hits:
        print("  (no results)")
        return
    for rank, h in enumerate(hits, start=1):
        meta = h["metadata"]
        snippet = " ".join(h["text"].split())
        snippet = snippet if len(snippet) <= width else snippet[:width] + "…"
        score = f"{h['score']:.4f}" if "score" in h else "  -   "
        print(f"  {rank}. [{score}] {meta.get('title', '')[:80]}")
        print(f"     {meta.get('url', '')}  ({meta.get('date', '')})")
        print(f"     {snippet}")


# ── Demo (steps 99 & 104) ────────────────────────────────────────
if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    question = sys.argv[2] if len(sys.argv) > 2 else f"What are {ticker}'s recent risks?"

    print(f"Fetching news for {ticker} from Tavily…")
    articles = fetch_news(ticker)
    print(f"  {len(articles)} articles fetched")

    n_whole = index_articles(articles, ticker, chunked=False)
    n_chunks = index_articles(articles, ticker, chunked=True)
    print(f"  Indexed {n_whole} whole-article docs → '{WHOLE_COLLECTION}'")
    print(f"  Indexed {n_chunks} chunks             → '{CHUNK_COLLECTION}'")

    print(f"\nQuestion: {question!r}")
    print("\n── Day 19: whole-article retrieval ──")
    print_hits(retrieve(question, ticker, k=3, collection=WHOLE_COLLECTION))
    print("\n── Day 20: chunked retrieval ──")
    print_hits(retrieve(question, ticker, k=3, collection=CHUNK_COLLECTION))
