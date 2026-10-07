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

Day 26: Multi-collection retrieval (news vs. SEC filings).
  129. ``retrieve(..., collection=...)`` accepts any collection name, or
       ``"auto"`` to route the question automatically.
  130. ``--route`` demo: a factual question against filings vs. a timely
       question against news.
  131. ``route_query()``: if the question mentions recency ("recent", "today",
       "latest", "news", "right now"…) → news; otherwise filings first, falling
       back to news when the filings match is weak or missing.

Collections:
    news         → one document per article (title + Tavily snippet)   [Day 19]
    news_chunks  → full article text split into overlapping chunks     [Day 20]
    filings      → SEC 10-Q MD&A + Risk Factors chunks                 [Day 25]

Usage:
    uv run python -m src.rag.retriever                  # NVDA, default query
    uv run python -m src.rag.retriever TSLA "What is driving Tesla's stock?"
    uv run python -m src.rag.retriever --route NVDA     # Day 26 routing demo
"""

from __future__ import annotations

import hashlib
import re
import sys
from typing import Any

from src.ingestion.tavily_client import filter_recent, search_news
from src.rag import vectorstore
from src.rag.chunker import chunk_text

WHOLE_COLLECTION = "news"  # Day 19
CHUNK_COLLECTION = "news_chunks"  # Day 20 — default from here on
NEWS_COLLECTION = CHUNK_COLLECTION  # alias used by the router
FILINGS_COLLECTION = "filings"  # Day 25 (see src.ingestion.edgar_client)
AUTO = "auto"
MAX_WORDS_PER_ARTICLE = 3000  # cap very long scraped pages (nav junk, comments…)

# Day 26: words that signal the user wants *timely* information → news.
RECENCY_TERMS = (
    "recent", "recently", "today", "latest", "news", "now", "currently",
    "this week", "yesterday", "breaking", "headline", "headlines",
)
_RECENCY_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in RECENCY_TERMS) + r")\b", re.I)
# Cosine similarity below which a filings match is considered "weak".
FILINGS_MIN_SCORE = 0.35


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


def route_query(query: str) -> list[str]:
    """
    Step 131: decide which collection(s) to search, in priority order.

    * Mentions recency ("recent", "today", "latest", "news", "now"…) → news only.
    * Anything else → filings first, news as the fallback.
    """
    if _RECENCY_RE.search(query):
        return [NEWS_COLLECTION]
    return [FILINGS_COLLECTION, NEWS_COLLECTION]


def has_documents(ticker: str, collection: str) -> bool:
    """True if *collection* holds at least one chunk for *ticker*."""
    res = vectorstore.get_collection(collection).get(
        where={"ticker": ticker.upper().strip()}, limit=1, include=[]
    )
    return bool(res["ids"])


def retrieve(
    query: str,
    ticker: str,
    k: int = 5,
    *,
    collection: str = CHUNK_COLLECTION,
    min_score: float = FILINGS_MIN_SCORE,
) -> list[dict[str, Any]]:
    """
    Step 98 / 129: semantic search over ONE ticker's stored documents.

    Args:
        collection: A collection name (``"news_chunks"``, ``"filings"``, …) or
            ``"auto"`` to let ``route_query`` choose. In auto mode, if the first
            collection's best match scores below *min_score* (or is empty), the
            next collection is searched too and results are merged by score.

    Returns:
        ``[{"id", "text", "metadata", "score"}, ...]`` best match first.
        In auto mode each hit's metadata gains ``"collection"``.
    """
    if collection != AUTO:
        return vectorstore.query(
            query,
            n_results=k,
            where={"ticker": ticker.upper().strip()},
            collection=collection,
        )

    merged: list[dict[str, Any]] = []
    for col in route_query(query):
        hits = retrieve(query, ticker, k, collection=col)
        for h in hits:
            h["metadata"]["collection"] = col
        merged.extend(hits)
        if hits and hits[0]["score"] >= min_score:
            break  # strong enough — no need to fall back
    return sorted(merged, key=lambda h: h["score"], reverse=True)[:k]


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


def _demo_routing(ticker: str) -> None:
    """Step 130: factual question vs. timely question, each against both collections."""
    for name, col in ((FILINGS_COLLECTION, FILINGS_COLLECTION), ("news", NEWS_COLLECTION)):
        if not has_documents(ticker, col):
            print(f"⚠️  No {name} stored for {ticker} — run "
                  f"{'`python -m src.ingestion.edgar_client`' if col == FILINGS_COLLECTION else '`python -m src.rag.retriever`'} first.")

    questions = [
        f"What was {ticker}'s revenue last quarter?",  # factual → filings
        f"What's the latest news on {ticker}?",  # timely → news
    ]
    for q in questions:
        print("=" * 78)
        print(f"Q: {q}\n   route_query → {route_query(q)}")
        for col in (FILINGS_COLLECTION, NEWS_COLLECTION):
            print(f"\n  ── forced collection = {col} ──")
            print_hits(retrieve(q, ticker, k=2, collection=col), width=160)
        print("\n  ── collection = auto ──")
        print_hits(retrieve(q, ticker, k=2, collection=AUTO), width=160)
    print("=" * 78)


# ── Demo (steps 99, 104 & 130) ───────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--route":
        _demo_routing(sys.argv[2].upper() if len(sys.argv) > 2 else "NVDA")
        sys.exit(0)

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
