"""
evaluation/evaluate.py
----------------------
Days 28 & 29: Automated evaluation harness for the golden test set.

Runs the 15 golden questions from evaluation/golden_dataset.json through the
StockSage RAG pipeline, checks retrieval accuracy against known key facts,
and prints a complete breakdown.

Usage:
    uv run python -m evaluation.evaluate
"""

import json
from pathlib import Path
from src.rag.hybrid import hybrid_retrieve
from src.rag.reranker import rerank
from src.rag.retriever import route_query, AUTO, FILINGS_COLLECTION, NEWS_COLLECTION

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def run_evaluation(candidates: int = 20, top_n: int = 4) -> dict:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []
    print("=" * 70)
    print(f"Running Golden Dataset Evaluation ({len(dataset)} questions)...")
    print(f"Parameters: candidates={candidates}, top_n={top_n}")
    print("=" * 70)

    for item in dataset:
        q = item["question"]
        ticker = item["ticker"]
        expected_src = item["expected_source"]
        key_facts = item["key_facts"]

        # Run routing + hybrid + rerank
        order = route_query(q)
        pool = hybrid_retrieve(q, ticker, k=candidates, collection=order[0])
        for h in pool:
            h["metadata"]["collection"] = order[0]

        top = rerank(q, pool, top_n=top_n)

        # Fallback to secondary collection if weak or empty
        if order[1:] and (not top or top[0]["score"] < 0.30):
            pool2 = hybrid_retrieve(q, ticker, k=candidates, collection=order[1])
            for h in pool2:
                h["metadata"]["collection"] = order[1]
            top = rerank(q, pool + pool2, top_n=top_n)

        retrieved_text = " ".join(d["text"] for d in top)
        retrieved_cols = [d["metadata"].get("collection") for d in top]

        facts_found = [f for f in key_facts if f.lower() in retrieved_text.lower()]
        has_majority = len(facts_found) >= max(1, (len(key_facts) + 1) // 2)
        has_source = expected_src in retrieved_cols
        passed = has_majority and has_source

        results.append({
            "id": item["id"],
            "ticker": ticker,
            "passed": passed,
            "facts_ratio": f"{len(facts_found)}/{len(key_facts)}",
            "source": retrieved_cols[0] if retrieved_cols else "none",
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{item['id']}] {status} | {ticker:4} | Facts: {len(facts_found)}/{len(key_facts)} | Source: {retrieved_cols[0] if retrieved_cols else 'none'}")

    passed_count = sum(1 for r in results if r["passed"])
    score_pct = (passed_count / len(dataset)) * 100
    print("-" * 70)
    print(f"Final Score: {passed_count}/{len(dataset)} ({score_pct:.1f}%)")
    print("=" * 70)
    return {"passed": passed_count, "total": len(dataset), "score_pct": score_pct}


if __name__ == "__main__":
    run_evaluation()
