"""
ingestion/explore_pandas.py
---------------------------
Steps 47–50: Practice pandas on AAPL price history.

  47. Load 6 months of AAPL price history into a DataFrame.
  48. Select 'Close', filter rows for the last 30 days, compute mean closing price.
  49. Handle missing data (NaN) with .dropna() or .fillna().
  50. Compute a simple 20-day moving average using .rolling(20).mean().

Usage:
    uv run python -m src.ingestion.explore_pandas
"""

import pandas as pd
import yfinance as yf


def main() -> None:
    # ── Step 47 ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 47: Load 6 months of AAPL price history")
    print("=" * 60)

    aapl = yf.Ticker("AAPL")
    df = aapl.history(period="6mo")

    print(f"  Shape:        {df.shape}")
    print(f"  Columns:      {list(df.columns)}")
    print(f"  Index name:   {df.index.name}")
    print(f"  Index dtype:  {df.index.dtype}")

    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(max_cols=8))

    print(f"\n  Last 3 rows:")
    print(df.tail(3).to_string(max_cols=8))

    # ── Step 48 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 48: Close column — last 30 days — mean")
    print("=" * 60)

    close = df["Close"]
    cutoff = df.index.max() - pd.Timedelta(days=30)
    last_30 = df.loc[df.index >= cutoff]
    mean_close = last_30["Close"].mean()

    print(f"  Close series length:  {len(close)}")
    print(f"  Last-30 window rows:  {len(last_30)}")
    print(f"  Date range:           {last_30.index.min().date()} → {last_30.index.max().date()}")
    print(f"  Mean closing price:   ${mean_close:.2f}")

    # ── Step 49 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 49: Handle missing data (NaN)")
    print("=" * 60)

    print(f"  NaN counts per column:")
    for col, count in df.isna().sum().items():
        print(f"    {col:>16s}  →  {count}")

    dropped = df.dropna(subset=["Close"])
    filled = df.ffill()
    print(f"\n  .dropna(subset=['Close']): {len(df)} → {len(dropped)} rows")
    print(f"  .ffill():                 {df.isna().sum().sum()} → {filled.isna().sum().sum()} total NaNs")

    df = dropped

    # ── Step 50 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Step 50: 20-day simple moving average")
    print("=" * 60)

    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    sma_nan_count = df["SMA_20"].isna().sum()
    print(f"  SMA_20 NaN rows (need 20 prior closes): {sma_nan_count}")
    print(f"  First valid SMA_20 date: {df['SMA_20'].first_valid_index()}")

    print(f"\n  Last 10 rows (Close vs SMA_20):")
    print(df[["Close", "SMA_20"]].tail(10).to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
