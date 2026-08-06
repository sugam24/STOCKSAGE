"""
ingestion/yfinance_client.py
----------------------------
Clean data-access layer for Yahoo Finance via ``yfinance``.

Steps covered:
  44. Two public functions: ``get_fundamentals`` and ``get_price_history``.
  45. Both functions return **validated Pydantic models** — never raw dicts.

Models
------
* ``CompanyFundamentals``  – core financial metrics sourced from ``.info``.
* ``PriceBar``            – a single day's OHLCV row.
* ``PriceHistory``        – wrapper around a list of ``PriceBar``'s.

Usage:
    uv run python -m src.ingestion.yfinance_client
"""

from datetime import date, datetime
from typing import Any

import yfinance as yf
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step 45 – Pydantic models
# ---------------------------------------------------------------------------

class CompanyFundamentals(BaseModel):
    """Validated snapshot of a company's key fundamental data."""

    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'.")
    short_name: str | None = Field(None, description="Human-readable company name.")
    sector: str | None = Field(None, description="GICS sector, e.g. 'Technology'.")
    industry: str | None = Field(None, description="GICS industry classification.")
    market_cap: int | None = Field(None, description="Market capitalisation in USD.")
    trailing_pe: float | None = Field(None, description="Trailing twelve-month P/E ratio.")
    forward_pe: float | None = Field(None, description="Forward (estimated) P/E ratio.")
    price_to_book: float | None = Field(None, description="Price-to-Book ratio.")
    dividend_yield: float | None = Field(None, description="Trailing dividend yield (decimal, not %).")
    beta: float | None = Field(None, description="Beta relative to the market.")
    fifty_two_week_high: float | None = Field(None, description="52-week high price.")
    fifty_two_week_low: float | None = Field(None, description="52-week low price.")
    currency: str | None = Field(None, description="Listing currency, e.g. 'USD'.")
    exchange: str | None = Field(None, description="Exchange code, e.g. 'NMS' (NASDAQ).")
    summary: str | None = Field(None, description="One-paragraph business summary.")


class PriceBar(BaseModel):
    """Single daily OHLCV bar."""

    trade_date: date = Field(..., alias="date", description="Trading date.")
    open_price: float = Field(..., alias="open", description="Opening price.")
    high: float = Field(..., description="Intraday high.")
    low: float = Field(..., description="Intraday low.")
    close: float = Field(..., description="Closing price.")
    volume: int = Field(..., description="Shares traded.")

    model_config = {"populate_by_name": True}


class PriceHistory(BaseModel):
    """Validated wrapper for a sequence of daily price bars."""

    ticker: str = Field(..., description="Stock ticker symbol.")
    period: str = Field(..., description="Requested look-back period, e.g. '6mo'.")
    bars: list[PriceBar] = Field(default_factory=list, description="Chronological list of daily OHLCV bars.")

    @property
    def trading_days(self) -> int:
        """Number of trading days in this history."""
        return len(self.bars)


# ---------------------------------------------------------------------------
# Step 44 – Public functions
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> CompanyFundamentals:
    """
    Fetch company fundamentals for *ticker* from Yahoo Finance.

    Returns a fully validated ``CompanyFundamentals`` Pydantic model.

    Args:
        ticker: Stock symbol (case-insensitive; normalised internally).

    Returns:
        CompanyFundamentals: Validated model with key financial metrics.

    Raises:
        ValueError: If the ticker is invalid or no data is returned.
    """
    ticker = ticker.upper().strip()
    info: dict = yf.Ticker(ticker).info

    if not info or info.get("regularMarketPrice") is None and info.get("shortName") is None:
        raise ValueError(
            f"No fundamental data returned for ticker '{ticker}'. "
            "Check that the symbol is valid."
        )

    return CompanyFundamentals(
        ticker=ticker,
        short_name=info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        trailing_pe=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        price_to_book=_safe_float(info.get("priceToBook")),
        dividend_yield=_safe_float(info.get("dividendYield")),
        beta=_safe_float(info.get("beta")),
        fifty_two_week_high=_safe_float(info.get("fiftyTwoWeekHigh")),
        fifty_two_week_low=_safe_float(info.get("fiftyTwoWeekLow")),
        currency=info.get("currency"),
        exchange=info.get("exchange"),
        summary=info.get("longBusinessSummary"),
    )


def get_price_history(ticker: str, period: str = "6mo") -> PriceHistory:
    """
    Fetch daily OHLCV price history for *ticker* over *period*.

    Valid periods: ``1d``, ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``,
    ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.

    Args:
        ticker: Stock symbol (case-insensitive; normalised internally).
        period: Look-back window accepted by ``yfinance`` (default ``"6mo"``).

    Returns:
        PriceHistory: Validated model containing chronological ``PriceBar``
        entries.

    Raises:
        ValueError: If the ticker is invalid or no history is returned.
    """
    ticker = ticker.upper().strip()
    hist = yf.Ticker(ticker).history(period=period)

    if hist.empty:
        raise ValueError(
            f"No price history returned for ticker '{ticker}' "
            f"with period='{period}'. Check that the symbol is valid."
        )

    bars: list[PriceBar] = []
    for idx, row in hist.iterrows():
        # idx is a pandas Timestamp — convert to date
        bar_date: date
        if isinstance(idx, datetime):
            bar_date = idx.date()
        else:
            bar_date = idx  # type: ignore[assignment]

        bars.append(
            PriceBar(
                trade_date=bar_date,
                open_price=round(float(row["Open"]), 4),
                high=round(float(row["High"]), 4),
                low=round(float(row["Low"]), 4),
                close=round(float(row["Close"]), 4),
                volume=int(row["Volume"]),
            )
        )

    return PriceHistory(ticker=ticker, period=period, bars=bars)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert *value* to float, returning None for missing / NaN values."""
    if value is None:
        return None
    try:
        f = float(value)
        # Guard against NaN (NaN != NaN is True)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Steps 44–45: yfinance_client.py demo")
    print("=" * 60)

    # ── get_fundamentals ────────────────────────────────────────────────
    print("\n📊  get_fundamentals('AAPL'):")
    fundamentals = get_fundamentals("AAPL")
    print(f"  Type: {type(fundamentals).__name__}")
    for field_name, field_value in fundamentals.model_dump().items():
        if field_name == "summary" and field_value:
            field_value = field_value[:100] + "…"
        print(f"  {field_name:>22s}: {field_value}")

    # ── get_price_history ───────────────────────────────────────────────
    print(f"\n📈  get_price_history('AAPL', period='6mo'):")
    history = get_price_history("AAPL", period="6mo")
    print(f"  Type:         {type(history).__name__}")
    print(f"  Ticker:       {history.ticker}")
    print(f"  Period:       {history.period}")
    print(f"  Trading days: {history.trading_days}")

    if history.bars:
        print(f"\n  First bar:  {history.bars[0].model_dump()}")
        print(f"  Last bar:   {history.bars[-1].model_dump()}")

    # ── Validate round-trip: model → dict → model ──────────────────────
    print("\n🔄  Round-trip validation:")
    raw_dict = fundamentals.model_dump()
    rebuilt = CompanyFundamentals.model_validate(raw_dict)
    assert rebuilt == fundamentals
    print("  ✅ CompanyFundamentals round-trip OK")

    raw_bar = history.bars[0].model_dump()
    rebuilt_bar = PriceBar.model_validate(raw_bar)
    assert rebuilt_bar == history.bars[0]
    print("  ✅ PriceBar round-trip OK")
