"""
ingestion/technical.py
----------------------
Steps 52–55: Technical indicators on price history.

  52. RSI formula documented in plain English (Wilder's method).
  53. ``compute_rsi(prices, period=14)`` — step-by-step pandas implementation.
  54. ``compute_moving_averages(prices)`` — latest MA20 and MA50.
  55. ``compute_technical_signals`` — combines both into a ``TechnicalSignals`` model.

Usage:
    uv run python -m src.ingestion.technical
"""

from typing import Sequence

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step 52 – RSI formula (Wilder's method, per Wikipedia / Investopedia)
# ---------------------------------------------------------------------------
#
# For each trading day, compare today's close to yesterday's close:
#   • If the close went UP, record the size of the gain (U); loss (D) is 0.
#   • If the close went DOWN, record the size of the loss (D); gain (U) is 0.
#   • If unchanged, both U and D are 0.
#
# Average the gains and losses over a look-back window (default 14 days)
# using Wilder's smoothed moving average (SMMA):
#   • The FIRST average is a simple mean of the first N gains / losses.
#   • Every subsequent average is:
#       new_avg = (previous_avg × (N − 1) + today's value) / N
#
# Relative Strength (RS) = average gain / average loss
# RSI = 100 − 100 / (1 + RS)          (ranges from 0 to 100)
#
# Interpretation: RSI above 70 → potentially overbought; below 30 → oversold.


# ---------------------------------------------------------------------------
# Step 55 – Pydantic model
# ---------------------------------------------------------------------------

class TechnicalSignals(BaseModel):
    """Validated snapshot of key technical indicators for a ticker."""

    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")
    latest_close: float | None = Field(None, description="Most recent closing price.")
    rsi: float | None = Field(None, description="Latest RSI value (14-period by default).")
    ma20: float | None = Field(None, description="Latest 20-day simple moving average.")
    ma50: float | None = Field(None, description="Latest 50-day simple moving average.")


# ---------------------------------------------------------------------------
# Step 53 – RSI
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series | Sequence[float], period: int = 14) -> pd.Series:
    """
    Compute Wilder's Relative Strength Index for a price series.

    Args:
        prices: Chronological closing prices (oldest first).
        period: Look-back window (default 14, per Wilder's recommendation).

    Returns:
        pd.Series of RSI values aligned with *prices*; the first *period*
        entries are NaN (insufficient history).
    """
    close = _to_series(prices)

    # Step 1: day-over-day price change
    delta = close.diff()

    # Step 2: split into gains (U) and losses (D)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Step 3: Wilder's SMMA — seed with a simple average, then smooth
    avg_gain = pd.Series(index=close.index, dtype=float)
    avg_loss = pd.Series(index=close.index, dtype=float)

    if len(close) <= period:
        return pd.Series(index=close.index, dtype=float)

    avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()

    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    # Step 4: RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Edge cases from Wilder's definition
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), other=50.0)
    rsi = rsi.where(~((avg_gain > 0) & (avg_loss == 0)), other=100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), other=0.0)

    return rsi


# ---------------------------------------------------------------------------
# Step 54 – Moving averages
# ---------------------------------------------------------------------------

def compute_moving_averages(
    prices: pd.Series | Sequence[float],
) -> tuple[float | None, float | None]:
    """
    Compute the latest 20-day and 50-day simple moving averages.

    Args:
        prices: Chronological closing prices (oldest first).

    Returns:
        Tuple of (MA20, MA50). Either value is ``None`` when there are
        fewer than the required number of observations.
    """
    close = _to_series(prices)

    ma20: float | None = None
    ma50: float | None = None

    if len(close) >= 20:
        ma20 = round(float(close.rolling(20).mean().iloc[-1]), 4)
    if len(close) >= 50:
        ma50 = round(float(close.rolling(50).mean().iloc[-1]), 4)

    return ma20, ma50


# ---------------------------------------------------------------------------
# Step 55 – Combine into validated model
# ---------------------------------------------------------------------------

def compute_technical_signals(
    prices: pd.Series | Sequence[float],
    ticker: str = "",
    rsi_period: int = 14,
) -> TechnicalSignals:
    """
    Compute RSI and moving averages from a price series.

    Args:
        prices: Chronological closing prices (oldest first).
        ticker: Stock symbol (stored on the returned model).
        rsi_period: RSI look-back window (default 14).

    Returns:
        Validated ``TechnicalSignals`` with the latest indicator values.
    """
    close = _to_series(prices)
    rsi_series = compute_rsi(close, period=rsi_period)
    ma20, ma50 = compute_moving_averages(close)

    latest_rsi = rsi_series.iloc[-1]
    rsi_value = None if pd.isna(latest_rsi) else round(float(latest_rsi), 4)

    latest_close = round(float(close.iloc[-1]), 4) if len(close) > 0 else None

    return TechnicalSignals(
        ticker=ticker.upper().strip() if ticker else "",
        latest_close=latest_close,
        rsi=rsi_value,
        ma20=ma20,
        ma50=ma50,
    )


def get_technical_signals(ticker: str, period: str = "6mo") -> TechnicalSignals:
    """
    Fetch price history from Yahoo Finance and compute technical signals.

    Args:
        ticker: Stock symbol (case-insensitive).
        period: Look-back window accepted by yfinance (default ``"6mo"``).

    Returns:
        Validated ``TechnicalSignals`` for the requested ticker.

    Raises:
        ValueError: If no price history is returned.
    """
    ticker = ticker.upper().strip()
    hist = yf.Ticker(ticker).history(period=period)

    if hist.empty:
        raise ValueError(
            f"No price history returned for ticker '{ticker}' "
            f"with period='{period}'. Check that the symbol is valid."
        )

    return compute_technical_signals(hist["Close"], ticker=ticker)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_series(prices: pd.Series | Sequence[float]) -> pd.Series:
    """Normalise *prices* to a float Series regardless of input type."""
    if isinstance(prices, pd.Series):
        return prices.astype(float).reset_index(drop=True)
    return pd.Series(prices, dtype=float)


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Steps 52–55: technical.py demo")
    print("=" * 60)

    signals = get_technical_signals("AAPL", period="6mo")
    print(f"\n  Type:         {type(signals).__name__}")
    for field_name, field_value in signals.model_dump().items():
        print(f"  {field_name:>14s}: {field_value}")

    print("\n  Round-trip validation:")
    rebuilt = TechnicalSignals.model_validate(signals.model_dump())
    assert rebuilt == signals
    print("  TechnicalSignals round-trip OK")
