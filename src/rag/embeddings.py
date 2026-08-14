"""
rag/embeddings.py
-----------------
Local text embeddings via sentence-transformers.

Steps covered:
  85. Install sentence-transformers (runs locally, completely free, no API key).
  86. Load the all-MiniLM-L6-v2 model and embed 3 sentences.
  87. Compute cosine similarity by hand using numpy — confirm similar sentences
      score higher.
  88. ``embed(texts: list[str]) -> list[list[float]]`` — production function.

Model : all-MiniLM-L6-v2  (384 dimensions, ~80 MB, CPU-friendly)

Usage:
    uv run python -m src.rag.embeddings
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the model so importing this module stays fast."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts into dense vectors.

    Parameters
    ----------
    texts : list[str]
        One or more strings to embed.

    Returns
    -------
    list[list[float]]
        A list of embedding vectors, each with 384 floats.
        Returned as plain Python lists for JSON serialisability.
    """
    if not texts:
        return []

    model = _get_model()
    # .encode() returns a numpy array of shape (n, 384)
    vectors = model.encode(texts)
    return vectors.tolist()


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # ── Step 86: Load model & embed 3 sentences ─────────────────
    sentences = [
        "Apple stock surged 5% after the earnings report.",       # finance
        "AAPL shares jumped following strong quarterly results.",  # finance (similar)
        "The weather of Kathmandu is too hot today",    # weather (different)
    ]

    vecs = embed(sentences)
    print(f"Model: all-MiniLM-L6-v2  |  Embedded {len(vecs)} texts → dim {len(vecs[0])}")

    # ── Step 87: Cosine similarity by hand with numpy ────────────
    def cosine_sim(a: list[float], b: list[float]) -> float:
        """cos(θ) = (a · b) / (‖a‖ × ‖b‖)"""
        a_, b_ = np.array(a), np.array(b)
        return float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_)))

    labels = ["finance-1", "finance-2", "weather"]

    print("\n── Pairwise cosine similarity ──")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_sim(vecs[i], vecs[j])
            print(f"  sim({labels[i]}, {labels[j]}) = {sim:.4f}")

    # Confirm similar sentences score higher
    sim_fin = cosine_sim(vecs[0], vecs[1])
    sim_mix1 = cosine_sim(vecs[0], vecs[2])
    sim_mix2 = cosine_sim(vecs[1], vecs[2])

    assert sim_fin > sim_mix1, "Finance pair should beat finance-vs-cooking!"
    assert sim_fin > sim_mix2, "Finance pair should beat finance-vs-cooking!"
    print("\n✅ Similar sentences score higher — embeddings work correctly.")
