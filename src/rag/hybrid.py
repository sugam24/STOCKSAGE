"""
rag/hybrid.py
-------------
Day 22: Hybrid search — embeddings + BM25 merged with Reciprocal Rank Fusion.

Steps covered:
  111. Run both retrievers on the same query → two ranked lists.
  112. Merge with RRF:  score(doc) = Σ_lists 1 / (60 + rank)   then re-sort.
  113. Demo compares hybrid vs. each retriever alone on 3 test questions.

Why RRF?  BM25 scores (unbounded, ~0-20) and cosine similarities (-1..1) live
on different scales, so you can't just add them. RRF ignores raw scores and
uses only *rank positions*, which makes it robust and parameter-light. The
constant 60 (from the original RRF paper) dampens the advantage of the very top
ranks so a doc that is #2 in both lists can beat one that is #1 in only one.

Usage:
    uv run python -m src.rag.hybrid [TICKER]
"""

from __future__ import annotations

import sys
from typing import Any

from src.rag.bm25 import bm25_search
from src.rag.retriever import CHUNK_COLLECTION, retrieve

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Merge several ranked result lists into one using RRF.

    Each input list holds ``{"id", "text", "metadata", ...}`` dicts, best first.
    Ranks are 1-based. Output dicts carry ``score`` (the fused RRF score) and
    ``sources`` — the per-retriever rank, e.g. ``{"list_0": 1, "list_1": 4}``.
    """
    fused: dict[str, dict[str, Any]] = {}
    for list_idx, results in enumerate(ranked_lists):
        for rank, doc in enumerate(results, start=1):
            entry = fused.setdefault(
                doc["id"],
                {"id": doc["id"], "text": doc["text"], "metadata": doc["metadata"],
                 "score": 0.0, "sources": {}},
            )
            entry["score"] += 1.0 / (k + rank)
            entry["sources"][f"list_{list_idx}"] = rank
    return sorted(fused.values(), key=lambda d: d["score"], reverse=True)


def hybrid_retrieve(
    query: str,
    ticker: str,
    k: int = 10,
    *,
    candidates: int = 20,
    collection: str = CHUNK_COLLECTION,
) -> list[dict[str, Any]]:
    """
    Hybrid search over one ticker's chunks.

    Args:
        query: Natural-language question.
        ticker: Restrict search to this ticker's stored news.
        k: Number of fused results to return.
        candidates: How many results to pull from *each* retriever before fusing.

    Returns:
        Top-*k* fused results; ``sources`` holds ``{"embedding": rank, "bm25": rank}``.
    """
    emb = retrieve(query, ticker, k=candidates, collection=collection)
    kw = bm25_search(query, ticker, k=candidates, collection=collection)
    fused = reciprocal_rank_fusion([emb, kw])
    for doc in fused:
        doc["sources"] = {
            name: doc["sources"][key]
            for key, name in (("list_0", "embedding"), ("list_1", "bm25"))
            if key in doc["sources"]
        }
    return fused[:k]


# ── Demo (step 113) ──────────────────────────────────────────────
if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    questions = [
        f"What are {ticker}'s recent risks?",
        "What did analysts say about the price target?",
        "competition from custom AI chips",
    ]

    def titles(hits: list[dict[str, Any]], n: int = 3) -> list[str]:
        return [
            f"{h['metadata'].get('title', '')[:60]} (chunk {h['metadata'].get('chunk_index')})"
            for h in hits[:n]
        ]

    for q in questions:
        print("=" * 78)
        print(f"Q: {q}")
        emb = retrieve(q, ticker, k=3)
        kw = bm25_search(q, ticker, k=3)
        hyb = hybrid_retrieve(q, ticker, k=3)
        for label, hits in (("Embedding", emb), ("BM25", kw)):
            print(f"\n  {label}:")
            for i, t in enumerate(titles(hits), 1):
                print(f"    {i}. {t}")
        print("\n  Hybrid (RRF):")
        for i, h in enumerate(hyb, 1):
            t = titles([h])[0]
            print(f"    {i}. [{h['score']:.4f}] {t}  ranks={h['sources']}")
    print("=" * 78)
