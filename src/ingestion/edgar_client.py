"""
ingestion/edgar_client.py
-------------------------
Day 25: SEC EDGAR — real company filings (10-Q / 10-K).

Steps covered:
  124. Fetch the most recent 10-Q for a ticker via SEC's free JSON APIs.
  125. Extract plain text and pull out the "Risk Factors" and "MD&A" sections.
  126. Chunk each section with the Day 20 chunker.
  127. Store the chunks in a separate ChromaDB collection called ``filings``.

How EDGAR works (no API key, but SEC *requires* a descriptive User-Agent):
  1. https://www.sec.gov/files/company_tickers.json      → ticker → CIK
  2. https://data.sec.gov/submissions/CIK##########.json → list of recent filings
  3. https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/<doc>
                                                         → the filing's HTML
SEC's fair-access policy: max 10 requests/second. We make 3 per filing.

Set ``SEC_USER_AGENT`` in .env ("YourApp your@email.com") to override the default.

Usage:
    uv run python -m src.ingestion.edgar_client            # NVDA
    uv run python -m src.ingestion.edgar_client AAPL 10-K
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "StockSage research dahalsugam24@gmail.com")
HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
TIMEOUT = 60

FILINGS_COLLECTION = "filings"

# Section boundaries by form type: (start heading regex, end heading regex).
# 10-Q: MD&A = Part I Item 2 → Item 3;  Risk Factors = Part II Item 1A → Item 2
# 10-K: MD&A = Item 7 → Item 7A;        Risk Factors = Item 1A → Item 1B / Item 2
_SECTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "10-Q": {
        "mdna": (r"item\s*2\.?\s*management[’'`s]*\s+discussion", r"item\s*3\.?\s*quantitative"),
        "risk_factors": (r"item\s*1a\.?\s*risk\s+factors", r"item\s*2\.?\s*unregistered|item\s*[3-6]\.?\s"),
    },
    "10-K": {
        "mdna": (r"item\s*7\.?\s*management[’'`s]*\s+discussion", r"item\s*7a\.?\s*quantitative"),
        "risk_factors": (r"item\s*1a\.?\s*risk\s+factors", r"item\s*1b\.?\s*unresolved|item\s*1c\.?\s|item\s*2\.?\s*properties"),
    },
}
SECTION_TITLES = {"mdna": "MD&A", "risk_factors": "Risk Factors"}


@dataclass
class Filing:
    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    url: str
    html: str = ""


# ── Step 124: locate & download the filing ───────────────────────
def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, dict[str, Any]]:
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    return {row["ticker"].upper(): row for row in data.values()}


def get_cik(ticker: str) -> tuple[int, str]:
    """Return ``(cik, company_name)`` for *ticker*."""
    row = _ticker_map().get(ticker.upper().strip())
    if row is None:
        raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers.json")
    return int(row["cik_str"]), row["title"]


def get_latest_filing(ticker: str, form: str = "10-Q", *, download: bool = True) -> Filing:
    """Find (and by default download) the most recent *form* filing for *ticker*."""
    ticker = ticker.upper().strip()
    cik, company = get_cik(ticker)
    recent = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]

    for i, f in enumerate(recent["form"]):
        if f == form:  # list is newest-first
            accession = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{accession.replace('-', '')}/{doc}"
            )
            filing = Filing(
                ticker=ticker,
                cik=cik,
                company=company,
                form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                accession=accession,
                url=url,
            )
            if download:
                filing.html = _get(url).text
            return filing
    raise ValueError(f"No {form} filing found in recent submissions for {ticker}")


# ── Step 125: HTML → text → sections ─────────────────────────────
def html_to_text(html: str) -> str:
    """Strip tags/scripts/hidden XBRL blocks and collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Inline-XBRL filings carry a hidden header block of machine-readable facts.
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_section(text: str, start_pattern: str, end_pattern: str) -> str:
    """
    Return the text between a section heading and the next section heading.

    A filing mentions each heading at least twice: once in the table of
    contents and once at the real section. We try every start match, pair it
    with the first end match after it, and keep the *longest* span — the TOC
    entry is only a few words long, the real section is thousands.
    """
    starts = [m.start() for m in re.finditer(start_pattern, text, re.I)]
    ends = [m.start() for m in re.finditer(end_pattern, text, re.I)]
    best = ""
    for s in starts:
        e = next((e for e in ends if e > s + 50), len(text))
        if e - s > len(best):
            best = text[s:e]
    return best.strip()


def extract_sections(text: str, form: str = "10-Q") -> dict[str, str]:
    """Extract ``{"mdna": ..., "risk_factors": ...}`` (empty string if not found)."""
    patterns = _SECTIONS.get(form, _SECTIONS["10-Q"])
    return {name: extract_section(text, s, e) for name, (s, e) in patterns.items()}


# ── Steps 126–127: chunk & store ─────────────────────────────────
def index_filing_sections(filing: Filing, sections: dict[str, str]) -> dict[str, int]:
    """Chunk already-extracted *sections* of *filing* and upsert them into ``filings``."""
    # Local imports keep this module importable without the RAG stack loaded.
    from src.rag import vectorstore
    from src.rag.chunker import chunk_text

    form = filing.form
    written: dict[str, int] = {}
    for name, body in sections.items():
        chunks = chunk_text(body)
        title = (
            f"{filing.company} {form} ({filing.report_date}) — {SECTION_TITLES[name]}"
        )
        ids = [f"{filing.accession}-{name}-{i}" for i in range(len(chunks))]
        docs = [f"{title}\n{c}" for c in chunks]
        metas = [
            {
                "ticker": filing.ticker,
                "source": "sec_edgar",
                "form": form,
                "section": name,
                "title": title,
                "url": filing.url,
                "accession": filing.accession,
                "date": filing.filing_date,
                "report_date": filing.report_date,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        written[name] = vectorstore.add(ids, docs, metas, collection=FILINGS_COLLECTION)
    return written


def index_filing(ticker: str, form: str = "10-Q") -> dict[str, int]:
    """
    Fetch the latest *form* filing, extract MD&A + Risk Factors, chunk them and
    upsert into the ``filings`` collection.

    Returns:
        ``{section_name: chunks_written}``.
    """
    filing = get_latest_filing(ticker, form)
    sections = extract_sections(html_to_text(filing.html), form)
    return index_filing_sections(filing, sections)


# ── Demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    form = sys.argv[2] if len(sys.argv) > 2 else "10-Q"

    f = get_latest_filing(ticker, form)
    print(f"{f.company} ({f.ticker}, CIK {f.cik})")
    print(f"  {f.form} filed {f.filing_date} for period ending {f.report_date}")
    print(f"  {f.url}")

    text = html_to_text(f.html)
    secs = extract_sections(text, form)
    print(f"\nFull filing text: {len(text.split()):,} words")
    for name, body in secs.items():
        words = body.split()
        print(f"\n── {SECTION_TITLES[name]}: {len(words):,} words ──")
        print("  " + " ".join(words[:60]) + " …")

    counts = index_filing(ticker, form)
    print(f"\nStored in '{FILINGS_COLLECTION}': {counts}")

    from src.rag.retriever import print_hits, retrieve

    q = f"What are the main risk factors for {ticker}?"
    print(f"\nTest query against '{FILINGS_COLLECTION}': {q!r}")
    print_hits(retrieve(q, ticker, k=3, collection=FILINGS_COLLECTION))
