"""
evaluation/build_snapshot.py
----------------------------
Day 27 helper: freeze the evaluation corpus.

Live news changes every day, so a golden dataset written against "whatever
Tavily returns now" would silently rot. This script captures, once:
  * the Tavily news articles (incl. full raw text) for each ticker, and
  * the MD&A + Risk Factors sections of each ticker's latest 10-Q,
into ``evaluation/corpus_snapshot.json``. The golden answers in
``golden_dataset.json`` were written by reading THIS snapshot, and
``run_eval.py`` indexes THIS snapshot into an isolated ChromaDB — so scores
are reproducible.

Usage (re-running overwrites the snapshot → golden answers must be re-checked):
    uv run python -m evaluation.build_snapshot NVDA AAPL TSLA
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.edgar_client import extract_sections, get_latest_filing, html_to_text
from src.rag.retriever import fetch_news

SNAPSHOT_PATH = Path(__file__).resolve().parent / "corpus_snapshot.json"
NEWS_FIELDS = ("title", "url", "published_date", "content", "raw_content")


def build(tickers: list[str]) -> dict:
    snapshot: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": {},
    }
    for t in tickers:
        t = t.upper()
        print(f"[{t}] fetching news…")
        news = [{k: a.get(k) for k in NEWS_FIELDS} for a in fetch_news(t)]
        print(f"[{t}] fetching latest 10-Q…")
        filing = get_latest_filing(t, "10-Q")
        sections = extract_sections(html_to_text(filing.html), "10-Q")
        meta = {k: v for k, v in asdict(filing).items() if k != "html"}
        snapshot["tickers"][t] = {"news": news, "filing": meta, "sections": sections}
        print(f"[{t}] {len(news)} articles, 10-Q {filing.report_date}, "
              f"sections: { {k: len(v.split()) for k, v in sections.items()} }")
    return snapshot


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["NVDA", "AAPL", "TSLA"]
    snap = build(tickers)
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=1, ensure_ascii=False))
    print(f"\nWrote {SNAPSHOT_PATH} ({SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")
