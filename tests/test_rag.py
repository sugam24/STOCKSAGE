"""Offline unit tests for the pure-logic parts of the RAG stack (no network, no models)."""

import pytest

from rag.bm25 import BM25Index, tokenize
from rag.chunker import chunk_text
from rag.hybrid import reciprocal_rank_fusion


# ── chunker ──────────────────────────────────────────────────────
def test_chunk_empty_text():
    assert chunk_text("   ") == []


def test_chunk_short_text_single_chunk():
    assert chunk_text("a b c", chunk_size=10, overlap=2) == ["a b c"]


def test_chunk_sizes_and_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_size=300, overlap=50)
    assert all(len(c.split()) <= 300 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-50:] == b.split()[:50]
    # every word is covered, last chunk ends with the last word
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w999"


def test_chunk_no_redundant_tail():
    # 300 words exactly → one chunk, not a second chunk made only of overlap
    assert len(chunk_text(" ".join(["x"] * 300), chunk_size=300, overlap=50)) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_chunk_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=size, overlap=overlap)


# ── BM25 ─────────────────────────────────────────────────────────
def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("What are NVDA's (recent) risks?") == ["what", "are", "nvda", "recent", "risks"]


def test_bm25_ranks_exact_term_match_first():
    docs = [
        {"id": "a", "text": "Apple launches a new iPhone", "metadata": {}},
        {"id": "b", "text": "NVDA faces export restrictions risks", "metadata": {}},
        {"id": "c", "text": "Oil prices fall on OPEC news", "metadata": {}},
    ]
    hits = BM25Index(docs).search("NVDA export risks", k=3)
    assert hits[0]["id"] == "b"
    assert all(h["score"] > 0 for h in hits)


def test_bm25_empty_index():
    assert BM25Index([]).search("anything") == []


# ── RRF ──────────────────────────────────────────────────────────
def _d(i):
    return {"id": i, "text": i, "metadata": {}}


def test_rrf_rewards_agreement():
    emb = [_d("a"), _d("b"), _d("c")]
    kw = [_d("b"), _d("d"), _d("a")]
    fused = reciprocal_rank_fusion([emb, kw])
    ids = [d["id"] for d in fused]
    # b: 1/62 + 1/61 ; a: 1/61 + 1/63 → b wins, both beat single-list docs
    assert ids[:2] == ["b", "a"]
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["sources"] == {"list_0": 2, "list_1": 1}
