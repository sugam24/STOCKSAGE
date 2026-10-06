"""
rag/bm25.py
-----------
Day 21: Keyword search with BM25 (Okapi BM25 via ``rank-bm25``).

Steps covered:
  106. Install rank-bm25.
  107. Build a BM25 index over the stored chunks (lowercase + whitespace split).
  108. Demo compares BM25 results to the embedding retriever on the same question.
  109. ``bm25_search(query, ticker)`` — clean wrapper with the same result shape
       as ``src.rag.retriever.retrieve``.

Why?  Embeddings capture *meaning* but can blur exact tokens such as tickers
("NVDA"), product names ("H200") or numbers ("$4.5B"). BM25 rewards documents
that literally contain the query terms, weighted by rarity (IDF) and
normalised for document length — a perfect complement to embeddings.

The index is built on demand from whatever is in ChromaDB for that ticker, so
it never goes stale relative to the vector store.

Usage:
    uv run python -m src.rag.bm25 [TICKER] ["question"]
"""

from __future__ import annotations

import string
import sys
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag import vectorstore
from src.rag.retriever import CHUNK_COLLECTION

_PUNCT = string.punctuation + "“”‘’—–…"


def tokenize(text: str) -> list[str]:
    """
    Lowercase + split on whitespace (step 107).

    Leading/trailing punctuation is stripped from each token so that
    ``"risks?"`` matches ``"risks"`` and ``"(NVDA)"`` matches ``"nvda"``.
    A trailing possessive ``'s`` is removed too (``"nvda's"`` → ``"nvda"``).
    """
    tokens: list[str] = []
    for raw in text.lower().split():
        tok = raw.strip(_PUNCT)
        if tok.endswith(("'s", "’s")):
            tok = tok[:-2]
        if tok:
            tokens.append(tok)
    return tokens


class BM25Index:
    """A BM25 index over a fixed list of ``{"id", "text", "metadata"}`` docs."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        corpus = [tokenize(d["text"]) for d in docs]
        # BM25Okapi raises on an empty corpus — guard for that.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-*k* docs by BM25 score (docs with score 0 are dropped)."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [
            {**self.docs[i], "score": float(scores[i])}
            for i in ranked[:k]
            if scores[i] > 0
        ]


def build_index(ticker: str, *, collection: str = CHUNK_COLLECTION) -> BM25Index:
    """Build a BM25 index over every stored chunk for *ticker*."""
    docs = vectorstore.get_documents(
        where={"ticker": ticker.upper().strip()},
        collection=collection,
    )
    return BM25Index(docs)


def bm25_search(
    query: str,
    ticker: str,
    k: int = 5,
    *,
    collection: str = CHUNK_COLLECTION,
) -> list[dict[str, Any]]:
    """Keyword-search one ticker's stored chunks. Same shape as ``retrieve()``."""
    return build_index(ticker, collection=collection).search(query, k=k)


# ── Demo (step 108) ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.rag.retriever import print_hits, retrieve

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    question = sys.argv[2] if len(sys.argv) > 2 else f"What are {ticker}'s recent risks?"

    index = build_index(ticker)
    print(f"BM25 index over {len(index.docs)} '{CHUNK_COLLECTION}' docs for {ticker}")
    if not index.docs:
        print("Nothing stored yet — run `uv run python -m src.rag.retriever` first.")
        sys.exit(1)

    print(f"\nQuestion: {question!r}   tokens={tokenize(question)}")
    print("\n── BM25 (keyword) ──")
    bm25_hits = index.search(question, k=3)
    print_hits(bm25_hits)
    print("\n── Embeddings (meaning) ──")
    emb_hits = retrieve(question, ticker, k=3)
    print_hits(emb_hits)

    overlap = {h["id"] for h in bm25_hits} & {h["id"] for h in emb_hits}
    print(f"\nOverlap between the two top-3 lists: {len(overlap)} doc(s)")