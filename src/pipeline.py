"""
pipeline.py
-----------
Phase 1 integration script — Step 76.

For a given ticker, this script:
  1. Fetches fundamentals + price history  (Day 8 – yfinance_client)
  2. Computes RSI and moving averages      (Day 10 – technical)
  3. Fetches recent news headlines          (Day 11 – tavily_client)
  4. Asks Groq to summarise everything     (Day 5 – structured output)
     into a single paragraph via a Pydantic-validated response.

Usage:
    uv run python -m src.pipeline              # runs AAPL, MSFT, NVDA
    uv run python -m src.pipeline TSLA GOOG    # custom tickers
"""

from __future__ import annotations

import sys
import time
import textwrap

from pydantic import BaseModel, Field

# ── Project imports ─────────────────────────────────────────────────────────
from src.ingestion.yfinance_client import get_fundamentals, get_price_history
from src.ingestion.technical import compute_technical_signals
from src.ingestion.tavily_client import search_news, filter_recent
from src.core.structured import extract


# ---------------------------------------------------------------------------
# Step 76 – Pydantic model for the one-paragraph summary
# ---------------------------------------------------------------------------

class TickerSummary(BaseModel):
    """One-paragraph research summary for a single stock ticker."""

    ticker: str = Field(..., description="The stock ticker symbol, e.g. 'AAPL'.")
    summary: str = Field(
        ...,
        description=(
            "A concise one-paragraph summary (3–5 sentences) covering the "
            "company's fundamentals, current technical signals (RSI, MAs), "
            "and recent news sentiment."
        ),
    )
    sentiment: str = Field(
        ...,
        description=(
            "Overall sentiment derived from the data: one of "
            "'bullish', 'bearish', or 'neutral'."
        ),
    )


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def research_ticker(ticker: str, *, verbose: bool = True) -> TickerSummary:
    """
    Run the full Phase 1 pipeline for *ticker* and return a validated
    ``TickerSummary`` from Groq.

    Steps:
      1. Fetch fundamentals via ``get_fundamentals(ticker)``.
      2. Fetch 6-month price history via ``get_price_history(ticker)``.
      3. Compute RSI/MA from the price bars.
      4. Fetch recent news via ``search_news(ticker)``.
      5. Build a context prompt and call ``extract()`` for structured output.

    Args:
        ticker:  Stock symbol (case-insensitive).
        verbose: Print progress to stdout.

    Returns:
        A validated ``TickerSummary`` Pydantic model.
    """
    ticker = ticker.upper().strip()

    # ── 1. Fundamentals ─────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  📊  Researching {ticker}")
        print(f"{'=' * 60}")
        print(f"  [1/5] Fetching fundamentals …")

    fundamentals = get_fundamentals(ticker)

    if verbose:
        print(f"        ✅ {fundamentals.short_name or ticker}  "
              f"| Sector: {fundamentals.sector or 'N/A'}  "
              f"| Mkt Cap: {_fmt_market_cap(fundamentals.market_cap)}")

    # ── 2. Price history ────────────────────────────────────────────────────
    if verbose:
        print(f"  [2/5] Fetching 6-month price history …")

    history = get_price_history(ticker, period="6mo")

    if verbose:
        print(f"        ✅ {history.trading_days} trading days retrieved")

    # ── 3. Technical signals ────────────────────────────────────────────────
    if verbose:
        print(f"  [3/5] Computing RSI & moving averages …")

    # Extract closing prices from PriceBar objects
    closing_prices = [bar.close for bar in history.bars]
    signals = compute_technical_signals(closing_prices, ticker=ticker)

    if verbose:
        print(f"        ✅ RSI: {signals.rsi}  |  MA20: {signals.ma20}  |  MA50: {signals.ma50}")

    # ── 4. News ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [4/5] Fetching recent news …")

    try:
        raw_news = search_news(ticker, max_results=5)
        recent_news = filter_recent(raw_news, max_age_days=30)
    except Exception as exc:
        if verbose:
            print(f"        ⚠️  News fetch failed ({exc}); continuing without news.")
        recent_news = []

    news_blurbs = _format_news(recent_news)

    if verbose:
        print(f"        ✅ {len(recent_news)} recent article(s)")

    # ── 5. Build prompt & extract structured summary ────────────────────────
    if verbose:
        print(f"  [5/5] Asking Groq to summarise …")

    prompt = _build_prompt(ticker, fundamentals, signals, news_blurbs)
    result: TickerSummary = extract(prompt, TickerSummary)

    if verbose:
        print(f"\n  🏁  Result for {ticker}:")
        print(f"  Sentiment : {result.sentiment}")
        print(f"  Summary   : {textwrap.fill(result.summary, width=72, initial_indent='    ', subsequent_indent='    ')}")
        print()

    return result


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    ticker: str,
    fundamentals,
    signals,
    news_blurbs: str,
) -> str:
    """Assemble all collected data into a single LLM prompt."""
    return textwrap.dedent(f"""\
        You are a senior equity analyst.  Summarise the following data for
        {ticker} into ONE concise paragraph (3–5 sentences).  Cover
        fundamentals, technical signals, and recent news sentiment.  Also
        classify overall sentiment as bullish, bearish, or neutral.

        === FUNDAMENTALS ===
        Company     : {fundamentals.short_name or ticker}
        Sector      : {fundamentals.sector or 'N/A'}
        Industry    : {fundamentals.industry or 'N/A'}
        Market Cap  : {_fmt_market_cap(fundamentals.market_cap)}
        Trailing PE : {fundamentals.trailing_pe or 'N/A'}
        Forward PE  : {fundamentals.forward_pe or 'N/A'}
        P/B         : {fundamentals.price_to_book or 'N/A'}
        Div Yield   : {_fmt_pct(fundamentals.dividend_yield)}
        Beta        : {fundamentals.beta or 'N/A'}
        52-wk High  : {fundamentals.fifty_two_week_high or 'N/A'}
        52-wk Low   : {fundamentals.fifty_two_week_low or 'N/A'}

        === TECHNICAL SIGNALS ===
        Latest Close : {signals.latest_close}
        RSI (14)     : {signals.rsi or 'N/A'}
        MA-20        : {signals.ma20 or 'N/A'}
        MA-50        : {signals.ma50 or 'N/A'}

        === RECENT NEWS ===
        {news_blurbs or 'No recent news available.'}
    """)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_market_cap(mc: int | None) -> str:
    if mc is None:
        return "N/A"
    if mc >= 1_000_000_000_000:
        return f"${mc / 1_000_000_000_000:.2f}T"
    if mc >= 1_000_000_000:
        return f"${mc / 1_000_000_000:.2f}B"
    if mc >= 1_000_000:
        return f"${mc / 1_000_000:.2f}M"
    return f"${mc:,}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_news(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = []
    for i, item in enumerate(articles[:5], start=1):
        title = item.get("title", "N/A")
        snippet = item.get("content", "")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        lines.append(f"{i}. {title}\n   {snippet}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point  (Step 77 — run for 3 tickers)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept tickers from the command line, or default to AAPL, MSFT, NVDA
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "NVDA"]

    print("╔" + "═" * 58 + "╗")
    print("║  StockSage — Phase 1 Integration Pipeline                ║")
    print("╚" + "═" * 58 + "╝")

    results: list[TickerSummary] = []
    for i, t in enumerate(tickers):
        if i > 0:
            time.sleep(2)  # be polite to APIs
        try:
            summary = research_ticker(t)
            results.append(summary)
        except Exception as exc:
            print(f"\n  ❌  Failed for {t.upper()}: {exc}\n")

    # ── Sanity-check recap ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📋  SANITY-CHECK RECAP")
    print("=" * 60)
    for r in results:
        print(f"\n  Ticker    : {r.ticker}")
        print(f"  Sentiment : {r.sentiment}")
        print(f"  Summary   : {textwrap.fill(r.summary, width=68, initial_indent='', subsequent_indent='              ')}")

    # Quick assertions for sanity
    print(f"\n{'─' * 60}")
    all_ok = True
    for r in results:
        checks = {
            "ticker non-empty": bool(r.ticker),
            "summary non-empty": bool(r.summary) and len(r.summary) > 20,
            "sentiment valid": r.sentiment.lower() in {"bullish", "bearish", "neutral"},
        }
        for label, passed in checks.items():
            status = "✅" if passed else "❌"
            if not passed:
                all_ok = False
            print(f"  {status}  {r.ticker} — {label}")

    print(f"\n  {'🎉 All checks passed!' if all_ok else '⚠️  Some checks failed — review above.'}")
    print()
