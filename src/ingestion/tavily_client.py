"""
ingestion/tavily_client.py
--------------------------
Steps 58–61: Tavily news search for a stock ticker.

  58. Requires TAVILY_API_KEY in .env (free tier: 1000 searches/month).
  59. ``search_news(ticker)`` queries Tavily for latest stock news.
  60. Demo prints title, URL, content snippet, and published date.
  61. Results older than 30 days are filtered out.

Usage:
    uv run python -m src.ingestion.tavily_client
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

MAX_AGE_DAYS = 30


def _get_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )
    return TavilyClient(api_key=api_key)


def _parse_published_date(raw: str | None) -> datetime | None:
    """Parse Tavily published_date strings into UTC-aware datetimes."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None


def search_news(ticker: str, *, max_results: int = 10) -> list[dict]:
    """
    Search Tavily for recent news about *ticker*.

    Args:
        ticker: Stock symbol (case-insensitive; normalised internally).
        max_results: Maximum number of results to request from Tavily.

    Returns:
        List of raw result dicts (title, url, content, published_date, …).

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    ticker = ticker.upper().strip()
    client = _get_client()

    response = client.search(
        query=f"latest news about {ticker} stock",
        topic="news",
        search_depth="basic",
        max_results=max_results,
    )
    return response.get("results", [])


def filter_recent(
    results: list[dict],
    *,
    max_age_days: int = MAX_AGE_DAYS,
) -> list[dict]:
    """
    Keep only results published within the last *max_age_days* days.

    Results with missing or unparseable ``published_date`` values are excluded.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    recent: list[dict] = []
    for item in results:
        published = _parse_published_date(item.get("published_date"))
        if published is not None and published >= cutoff:
            recent.append(item)
    return recent


def print_results(results: list[dict]) -> None:
    """Print title, URL, content snippet, and published date for each result."""
    if not results:
        print("No results to display.")
        return

    for i, item in enumerate(results, start=1):
        title = item.get("title", "N/A")
        url = item.get("url", "N/A")
        snippet = item.get("content", "")
        published = item.get("published_date", "N/A")

        print(f"\n--- Result {i} ---")
        print(f"Title:     {title}")
        print(f"URL:       {url}")
        print(f"Published: {published}")
        if len(snippet) > 300:
            print(f"Snippet:   {snippet[:300]}…")
        else:
            print(f"Snippet:   {snippet}")


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_ticker = "AAPL"

    print("=" * 60)
    print(f"  Steps 58–61: Tavily news for {demo_ticker}")
    print("=" * 60)

    all_results = search_news(demo_ticker)
    print(f"\nRaw results from Tavily: {len(all_results)}")

    recent_results = filter_recent(all_results, max_age_days=MAX_AGE_DAYS)
    print(f"Results within last {MAX_AGE_DAYS} days: {len(recent_results)}")

    print_results(recent_results)
