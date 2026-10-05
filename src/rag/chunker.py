"""
rag/chunker.py
--------------
Day 20: Split long documents into overlapping chunks before embedding.

Steps covered:
  101. ``chunk_text()`` splits text into ~300-word chunks with a 50-word
       overlap so context isn't lost at chunk boundaries.
  102. Demo below runs it on a full article's worth of text.

Why chunk?  One embedding for a 2,000-word article averages together many
different ideas (earnings, lawsuits, product launches...) and becomes a blurry
match for everything. Smaller chunks each capture one or two ideas, so a
question about "risks" lands on the paragraph that actually discusses risks.

Usage:
    uv run python -m src.rag.chunker
"""

from __future__ import annotations

DEFAULT_CHUNK_SIZE = 300  # words
DEFAULT_OVERLAP = 50  # words


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split *text* into word-based chunks with overlap.

    Each chunk has at most *chunk_size* words; consecutive chunks share
    *overlap* words. A trailing chunk that would contain only words already
    covered by the previous chunk's overlap is not emitted.

    Args:
        text: The document to split.
        chunk_size: Max words per chunk (must be > 0).
        overlap: Words shared between consecutive chunks (0 <= overlap < chunk_size).

    Returns:
        List of chunk strings (empty list for blank input).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


# ── Demo (step 102) ──────────────────────────────────────────────
if __name__ == "__main__":
    paragraphs = [
        "Nvidia reported quarterly revenue that beat Wall Street estimates, driven by "
        "relentless demand for its data-center GPUs used to train and run large AI models. "
        "Data-center revenue rose sharply year over year, and the company guided above "
        "consensus for the next quarter.",
        "However, analysts flagged several risks. U.S. export restrictions on advanced chips "
        "to China could weigh on sales, and the company has already taken charges related to "
        "inventory it can no longer ship. Customer concentration is another concern: a handful "
        "of hyperscalers account for a large share of revenue.",
        "Competition is intensifying as AMD ramps its MI-series accelerators and cloud giants "
        "design their own custom silicon. Some investors worry that gross margins may come "
        "under pressure as the product mix shifts and new architectures ramp.",
        "On the supply side, Nvidia depends heavily on TSMC for manufacturing and on a small "
        "number of suppliers for high-bandwidth memory. Any disruption, including geopolitical "
        "tension around Taiwan, could constrain shipments.",
        "Despite these concerns, management struck an upbeat tone, highlighting new product "
        "cycles, software revenue, and growing adoption in automotive and robotics.",
    ]
    # Repeat to reach a realistic full-article length (~1,000+ words)
    article = "\n\n".join(paragraphs * 4)

    chunks = chunk_text(article)
    print(f"Article length : {len(article.split())} words")
    print(f"Chunks         : {len(chunks)} (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_OVERLAP})")
    for i, c in enumerate(chunks):
        w = c.split()
        print(f"\n[{i}] {len(w)} words | starts: {' '.join(w[:8])}… | ends: …{' '.join(w[-8:])}")

    # Verify overlap: last 50 words of chunk i == first 50 words of chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-DEFAULT_OVERLAP:] == b.split()[:DEFAULT_OVERLAP]
    print("\n✅ Consecutive chunks share exactly", DEFAULT_OVERLAP, "words of overlap.")
