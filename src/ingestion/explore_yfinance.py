"""
ingestion/explore_yfinance.py
-----------------------------
Steps 41–43: Explore the yfinance API.

  41. Install yfinance, fetch a Ticker object for AAPL.
  42. Explore .info — company fundamentals (P/E, market cap, sector …).
  43. Explore .history(period='6mo') — pandas DataFrame of daily OHLCV data.

Usage:
    uv run python -m src.ingestion.explore_yfinance
"""

import yfinance as yf


def main() -> None:
    # ── Step 41 ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 41: yfinance installed")
    print("=" * 60)
    print(f"  Version: {yf.__version__}\n")

    aapl = yf.Ticker("AAPL")
    print(f"  Ticker object: {aapl}")
    print(f"  Type: {type(aapl)}\n")

    # ── Step 42 ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 42: .info — company fundamentals")
    print("=" * 60)
    info = aapl.info

    fundamentals = [
        "shortName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "beta",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "averageVolume",
        "currency",
        "exchange",
    ]

    for key in fundamentals:
        value = info.get(key, "N/A")
        print(f"  {key:>22s}: {value}")

    summary = info.get("longBusinessSummary", "")
    if summary:
        print(f"\n  longBusinessSummary (first 150 chars):\n    {summary[:150]}…")

    print(f"\n  Total keys in .info dict: {len(info)}")
    print(f"  All keys (first 30): {sorted(info.keys())[:30]}")

    # ── Step 43 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 43: .history(period='6mo') — OHLCV DataFrame")
    print("=" * 60)
    hist = aapl.history(period="6mo")

    print(f"  Shape:        {hist.shape}")
    print(f"  Columns:      {list(hist.columns)}")
    print(f"  Index name:   {hist.index.name}")
    print(f"  Index dtype:  {hist.index.dtype}")
    print(f"\n  Column dtypes:")
    for col in hist.columns:
        print(f"    {col:>16s}  →  {hist[col].dtype}")

    print(f"\n  First 3 rows:")
    print(hist.head(3).to_string(max_cols=8))

    print(f"\n  Last 3 rows:")
    print(hist.tail(3).to_string(max_cols=8))


if __name__ == "__main__":
    main()
