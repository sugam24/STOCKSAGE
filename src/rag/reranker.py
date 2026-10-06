"""
rag/reranker.py
---------------
Day 23: Rerank hybrid-search candidates with FlashRank.

Steps covered:
  115. Install flashrank (free, runs locally on CPU, no API key).
  116. Rerank the Day 22 hybrid results against the original query.
  117. Keep only the top 3.
  118. ``rerank(query, docs, top_n=3)`` wrapper.

Why rerank?  Retrievers (bi-encoders, BM25) score the query and each document
*independently*, which is fast but coarse. A reranker is a cross-encoder: it
reads the query and a candidate *together* and outputs one relevance score, so
it's much more precise — but too slow to run over the whole corpus. Hence the
classic pattern: cheap retrieval for ~10-20 candidates → precise rerank → top 3.

Model: ms-marco-TinyBERT-L-2-v2 (FlashRank default, ~4 MB, cached in data/flashrank).

Usage:
    uv run python -m src.rag.reranker [TICKER] ["question"]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from flashrank import Ranker, RerankRequest

MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "flashrank"
DEFAULT_TOP_N = 3

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """Lazy-load the reranker (first call downloads the model into CACHE_DIR)."""
    global _ranker
    if _ranker is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _ranker = Ranker(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR), log_level="WARNING")
    return _ranker


def rerank(
    query: str,
    docs: list[dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """
    Re-score *docs* against *query* and keep the *top_n* most relevant.

    Args:
        query: The original user question.
        docs: Candidates shaped ``{"id", "text", "metadata", "score", ...}``
              (e.g. output of ``hybrid_retrieve``).
        top_n: How many to keep.

    Returns:
        The best *top_n* docs, each a copy of the input dict with ``score``
        replaced by the reranker's relevance score and the previous score kept
        under ``retrieval_score``.
    """
    if not docs:
        return []

    by_id = {d["id"]: d for d in docs}
    passages = [{"id": d["id"], "text": d["text"]} for d in docs]
    results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))

    reranked: list[dict[str, Any]] = []
    for r in results[:top_n]:
        original = by_id[r["id"]]
        reranked.append(
            {**original, "retrieval_score": original.get("score"), "score": float(r["score"])}
        )
    return reranked


# ── Demo (steps 116–117) ─────────────────────────────────────────
if __name__ == "__main__":
    from src.rag.hybrid import hybrid_retrieve

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    question = sys.argv[2] if len(sys.argv) > 2 else f"What are {ticker}'s recent risks?"

    candidates = hybrid_retrieve(question, ticker, k=10)
    print(f"Q: {question}\n\n── Hybrid top-10 (before rerank) ──")
    for i, c in enumerate(candidates, 1):
        print(f"  {i:2}. [{c['score']:.4f}] {c['metadata'].get('title', '')[:60]} "
              f"(chunk {c['metadata'].get('chunk_index')})")

    top = rerank(question, candidates)
    print(f"\n── FlashRank top-{DEFAULT_TOP_N} ──")
    for i, c in enumerate(top, 1):
        snippet = " ".join(c["text"].split())[:220]
        print(f"  {i}. [{c['score']:.4f}] {c['metadata'].get('title', '')[:60]} "
              f"(chunk {c['metadata'].get('chunk_index')})\n     {snippet}…")
