"""
rag/vectorstore.py
------------------
Day 18: ChromaDB — a free, local vector database.

Steps covered:
  90. Install chromadb and create a *persistent* local client (data lives in
      ``data/chroma/`` and survives between runs).
  91. Create a collection called ``news``.
  92. Add 5 sample sentences using our Day 17 ``embed()`` vectors.
  93. Query with a new sentence and inspect the closest matches.
  94. ``add()`` and ``query()`` wrappers used by the rest of the RAG stack.

We always pass our *own* embeddings (from ``src.rag.embeddings``) so Chroma
never silently uses its built-in embedding model. Collections are created with
cosine distance, so ``score = 1 - distance`` is a cosine similarity in [-1, 1].

Usage:
    uv run python -m src.rag.vectorstore
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.rag.embeddings import embed

# Project root → <root>/data/chroma  (git-ignored)
PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
DEFAULT_COLLECTION = "news"

_client: Any = None  # chromadb.PersistentClient instance (lazy)


def get_client() -> Any:
    """Return a lazily-created persistent Chroma client."""
    global _client
    if _client is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def get_collection(name: str = DEFAULT_COLLECTION) -> Collection:
    """Get (or create) a collection that uses cosine distance."""
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _clean_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma only accepts str/int/float/bool values — drop None, stringify the rest."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


def add(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    *,
    collection: str = DEFAULT_COLLECTION,
) -> int:
    """
    Embed *documents* and upsert them into *collection*.

    Upsert (instead of add) makes re-indexing idempotent: re-running with the
    same ids overwrites rather than duplicating.

    Returns:
        Number of documents written.
    """
    if not documents:
        return 0
    if len(ids) != len(documents):
        raise ValueError("ids and documents must have the same length")
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("metadatas and documents must have the same length")

    col = get_collection(collection)
    col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embed(documents),  # type: ignore[arg-type]
        metadatas=[_clean_metadata(m) for m in metadatas] if metadatas else None,  # type: ignore[arg-type]
    )
    return len(documents)


def query(
    text: str,
    n_results: int = 5,
    *,
    where: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """
    Search *collection* by meaning.

    Args:
        text: The natural-language query.
        n_results: How many matches to return.
        where: Optional metadata filter, e.g. ``{"ticker": "NVDA"}``.
        collection: Collection name.

    Returns:
        List of ``{"id", "text", "metadata", "score"}`` dicts, best match first.
        ``score`` is cosine similarity (higher = more similar).
    """
    col = get_collection(collection)
    if col.count() == 0:
        return []

    res = col.query(
        query_embeddings=embed([text]),  # type: ignore[arg-type]
        n_results=min(n_results, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
    )

    ids = res["ids"][0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    return [
        {
            "id": ids[i],
            "text": docs[i],
            "metadata": dict(metas[i] or {}),
            "score": 1.0 - float(dists[i]),
        }
        for i in range(len(ids))
    ]


def get_documents(
    *,
    where: dict[str, Any] | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> list[dict[str, Any]]:
    """Return every stored document (optionally filtered) — used by BM25."""
    col = get_collection(collection)
    res = col.get(where=where, include=["documents", "metadatas"])  # type: ignore[list-item]
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        {"id": res["ids"][i], "text": docs[i], "metadata": dict(metas[i] or {})}
        for i in range(len(res["ids"]))
    ]


def delete_collection(name: str) -> None:
    """Drop a collection entirely (no-op if it doesn't exist)."""
    try:
        get_client().delete_collection(name)
    except Exception:  # noqa: BLE001 — Chroma raises different types across versions
        pass


# ── Demo (steps 91–93) ───────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Nvidia reports record data-center revenue driven by AI chip demand.",
        "Apple unveils a new iPhone with an upgraded camera system.",
        "The Federal Reserve holds interest rates steady amid inflation concerns.",
        "Tesla recalls thousands of vehicles over a software issue.",
        "Oil prices fall as OPEC signals higher production next quarter.",
    ]
    ids = [f"sample-{i}" for i in range(len(samples))]
    metas = [{"ticker": "DEMO", "source": "sample"} for _ in samples]

    written = add(ids, samples, metas, collection=DEFAULT_COLLECTION)
    print(f"Persist dir : {PERSIST_DIR}")
    print(f"Upserted {written} sample sentences into '{DEFAULT_COLLECTION}' "
          f"(collection now has {get_collection().count()} docs)")

    for q in [
        "GPU maker sees booming artificial intelligence sales",
        "central bank monetary policy decision",
    ]:
        print(f"\nQuery: {q!r}")
        for hit in query(q, n_results=3, where={"source": "sample"}):
            print(f"  {hit['score']:.4f}  {hit['text']}")
