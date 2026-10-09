"""
agents/risk_metrics.py
-----------------------
Day 37: From-scratch quantitative risk calculations.

Implements core risk metrics from daily price histories:
  1. Annualized Volatility (standard deviation of daily returns * sqrt(252))
  2. Historical Value at Risk (VaR at confidence level, default 95%)
  3. Maximum Drawdown (peak-to-trough percentage drop)

Usage:
    uv run python -m src.agents.risk_metrics
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from pydantic import BaseModel, Field


class RiskMetricsSnapshot(BaseModel):
    """Structured container for quantitative risk metrics."""

    volatility_annualized: float = Field(..., description="Annualized volatility (sigma * sqrt(252)).")
    var_95_daily: float = Field(..., description="Historical 1-day Value at Risk at 95% confidence.")
    max_drawdown: float = Field(..., description="Maximum peak-to-trough drop (negative decimal).")
    sample_bars: int = Field(..., description="Number of daily price observations analyzed.")


def _extract_prices(price_history: Sequence[float] | Any) -> list[float]:
    """Extract a list of float prices from either a raw sequence or a PriceHistory model."""
    if hasattr(price_history, "bars"):
        return [float(b.close) for b in price_history.bars]
    if isinstance(price_history, (list, tuple)):
        return [float(p) for p in price_history]
    raise TypeError(f"Unsupported price_history type: {type(price_history)}")


def compute_daily_returns(prices: Sequence[float]) -> list[float]:
    """Compute simple daily percentage returns: r_t = (P_t - P_{t-1}) / P_{t-1}."""
    if len(prices) < 2:
        return []
    returns: list[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev <= 0:
            continue
        returns.append((curr - prev) / prev)
    return returns


def compute_volatility(
    price_history: Sequence[float] | Any,
    trading_days: int = 252,
) -> float:
    """
    Step 174: Compute annualized volatility from price history.

    Formula:
        Daily stddev = sqrt( sum((r - mean)^2) / (N - 1) )
        Annualized vol = Daily stddev * sqrt(trading_days)
    """
    prices = _extract_prices(price_history)
    returns = compute_daily_returns(prices)
    n = len(returns)

    if n < 2:
        return 0.0

    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
    daily_std = math.sqrt(variance)
    return float(daily_std * math.sqrt(trading_days))


def compute_var(
    price_history: Sequence[float] | Any,
    confidence: float = 0.95,
) -> float:
    """
    Step 175: Compute 1-day Value at Risk using historical simulation.

    Sorts historical daily returns and extracts the (1 - confidence) percentile.
    Reported as a positive loss percentage (e.g. 0.035 = 3.5% max expected 1-day loss).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")

    prices = _extract_prices(price_history)
    returns = compute_daily_returns(prices)
    if not returns:
        return 0.0

    sorted_returns = sorted(returns)
    alpha = 1.0 - confidence
    rank = alpha * (len(sorted_returns) - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    weight = rank - lower_idx

    # Linear interpolation between adjacent ranks
    cutoff_return = (
        sorted_returns[lower_idx] * (1.0 - weight)
        + sorted_returns[upper_idx] * weight
    )

    # VaR represents loss magnitude: if return is -0.04, VaR is 0.04
    loss = -cutoff_return
    return float(max(0.0, loss))


def compute_max_drawdown(price_history: Sequence[float] | Any) -> float:
    """
    Step 176: Compute maximum peak-to-trough drawdown across price history.

    Returns:
        Negative float (e.g. -0.22 = 22% maximum drawdown).
    """
    prices = _extract_prices(price_history)
    if len(prices) < 2:
        return 0.0

    peak = prices[0]
    max_dd = 0.0

    for p in prices:
        if p > peak:
            peak = p
        elif peak > 0:
            dd = (p - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return float(max_dd)


def compute_all_risk_metrics(price_history: Sequence[float] | Any) -> RiskMetricsSnapshot:
    """Compute all three core risk metrics in a single pass."""
    prices = _extract_prices(price_history)
    return RiskMetricsSnapshot(
        volatility_annualized=round(compute_volatility(prices), 4),
        var_95_daily=round(compute_var(prices, confidence=0.95), 4),
        max_drawdown=round(compute_max_drawdown(prices), 4),
        sample_bars=len(prices),
    )


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 37: Risk Metrics Demo")
    print("=" * 65)

    # Example: 5-day price trajectory
    sample_prices = [100.0, 105.0, 95.0, 102.0, 98.0]
    print(f"Sample prices: {sample_prices}")
    rets = compute_daily_returns(sample_prices)
    print(f"Daily returns: {[round(r, 4) for r in rets]}")

    snapshot = compute_all_risk_metrics(sample_prices)
    print(f"\nCalculated Risk Metrics:")
    print(f"  • Annualized Volatility: {snapshot.volatility_annualized:.2%}")
    print(f"  • 95% 1-Day Historical VaR: {snapshot.var_95_daily:.2%}")
    print(f"  • Maximum Drawdown: {snapshot.max_drawdown:.2%}")
    print(f"  • Sample Bars: {snapshot.sample_bars}")
