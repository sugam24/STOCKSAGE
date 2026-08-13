# Retrieval-Augmented Generation (RAG)

## What Is RAG?

RAG is a technique that makes an LLM smarter about *your* data by fetching relevant documents at query-time and injecting them into the prompt, rather than relying solely on whatever the model memorised during pre-training. When a user asks a question, the system first converts that question into a numerical vector (an embedding), then searches a vector store for the document chunks whose embeddings are most similar — these are the "top matches." Those top matches are stitched into the prompt alongside the original question, so the LLM (Groq's Llama 3.3 in our case) can ground its answer in real, specific evidence instead of hallucinating facts. This means the model's knowledge is no longer frozen at its training cutoff; we can update, add, or remove documents at any time and the answers change immediately without retraining. In the context of StockSage, RAG will let us feed earnings reports, SEC filings, and analyst notes into a vector database so the assistant can answer detailed, source-backed financial questions that no general-purpose LLM could handle on its own.

---

## RAG Flow Diagram

The diagram below traces a single user question through the entire RAG pipeline
from start to finish.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │   QUESTION   │  "What was AAPL's revenue guidance for Q3?"
  │  (raw text)  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────┐
  │  EMBED QUESTION  │  Convert the question into a numerical vector
  │  (embedding      │  using an embedding model (e.g. all-MiniLM-L6)
  │   model)         │
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ SEARCH DOCUMENTS │  Perform a similarity search against the
  │  (vector store)  │  vector database (ChromaDB / FAISS / etc.)
  │                  │  to find chunks closest to the question vector
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │   TOP MATCHES    │  Return the k most relevant document chunks
  │  (k chunks)      │  ranked by cosine similarity
  │                  │
  │  ┌────────────┐  │
  │  │ chunk 1    │  │  ← "AAPL reported revenue guidance of..."
  │  │ chunk 2    │  │  ← "Management noted Q3 expectations..."
  │  │ chunk 3    │  │  ← "Analysts revised targets after..."
  │  └────────────┘  │
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │            GROQ  +  MATCHES                  │
  │                                              │
  │  Prompt to LLM (llama-3.3-70b-versatile):   │
  │  ┌────────────────────────────────────────┐  │
  │  │ SYSTEM: You are a financial analyst.  │  │
  │  │                                        │  │
  │  │ CONTEXT (retrieved chunks):            │  │
  │  │   • chunk 1                            │  │
  │  │   • chunk 2                            │  │
  │  │   • chunk 3                            │  │
  │  │                                        │  │
  │  │ USER QUESTION:                         │  │
  │  │   "What was AAPL's revenue guidance    │  │
  │  │    for Q3?"                            │  │
  │  └────────────────────────────────────────┘  │
  └──────────────┬───────────────────────────────┘
                 │
                 ▼
  ┌──────────────────┐
  │     ANSWER       │  "According to Apple's Q3 earnings call,
  │  (grounded in    │   management guided for revenue between
  │   real docs)     │   $89.5B and $93.5B, citing strong demand
  │                  │   in Services and iPhone segments."
  └──────────────────┘


 ═══════════════════════════════════════════════════════════════
  LEGEND
  ─────
  Embedding Model   →  Converts text → dense vector (e.g. 384-dim)
  Vector Store      →  Database of pre-embedded document chunks
  Similarity Search →  Cosine similarity between query & doc vectors
  Groq (LLM)        →  Generates the final answer using context
 ═══════════════════════════════════════════════════════════════
```

---

## Why RAG Matters for StockSage

| Without RAG | With RAG |
|---|---|
| LLM only knows its training data (stale) | LLM reads fresh earnings reports, filings |
| Answers are generic, may hallucinate numbers | Answers cite specific documents and figures |
| Cannot reference private/proprietary data | Can ingest any PDF, HTML, or text you own |
| Requires fine-tuning to add new knowledge | Just add new docs to the vector store |

---

## StockSage RAG Roadmap (Phase 2)

1. **Embed & Store** — Chunk financial documents, embed them, store in a vector DB
2. **Retrieve** — At query time, find the top-k most relevant chunks
3. **Augment** — Inject those chunks into the Groq prompt as context
4. **Generate** — Let the LLM synthesise a grounded, source-backed answer
5. **Evaluate** — Measure retrieval quality (recall@k) and answer faithfulness

> **Key takeaway:** RAG separates *what the model knows* from *what it can access*.
> The model's reasoning ability stays the same, but its knowledge becomes dynamic
> and controllable.
