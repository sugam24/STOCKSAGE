"""
tests/test_risk_metrics.py
---------------------------
Unit tests for Day 37 quantitative risk metrics (Volatility, VaR, Max Drawdown).
"""

import math
import pytest
from src.agents.risk_metrics import (
    compute_all_risk_metrics,
    compute_daily_returns,
    compute_max_drawdown,
    compute_var,
    compute_volatility,
)


def test_compute_daily_returns_hand_calculable():
    """
    Given prices: [100.0, 110.0, 88.0, 96.8]
    Returns should be exactly: [+0.10, -0.20, +0.10].
    """
    prices = [100.0, 110.0, 88.0, 96.8]
    rets = compute_daily_returns(prices)
    assert len(rets) == 3
    assert pytest.approx(rets[0], rel=1e-5) == 0.10
    assert pytest.approx(rets[1], rel=1e-5) == -0.20
    assert pytest.approx(rets[2], rel=1e-5) == 0.10


def test_compute_volatility_hand_calculable():
    """
    For returns [+0.10, -0.20, +0.10]:
      mean = 0.0
      sum((r - mean)^2) = 0.01 + 0.04 + 0.01 = 0.06
      variance (N-1=2) = 0.06 / 2 = 0.03
      std = sqrt(0.03) ≈ 0.173205
      annualized (252 days) = sqrt(0.03) * sqrt(252) ≈ 2.749545
    """
    prices = [100.0, 110.0, 88.0, 96.8]
    vol = compute_volatility(prices)
    expected_vol = math.sqrt(0.03) * math.sqrt(252)
    assert pytest.approx(vol, rel=1e-4) == expected_vol


def test_compute_max_drawdown_hand_calculable():
    """
    For prices [100.0, 110.0, 88.0, 96.8]:
      Peak reaches 110.0, lowest point following peak is 88.0.
      Drawdown = (88.0 - 110.0) / 110.0 = -22.0 / 110.0 = -0.20 (-20%).
    """
    prices = [100.0, 110.0, 88.0, 96.8]
    mdd = compute_max_drawdown(prices)
    assert pytest.approx(mdd, rel=1e-5) == -0.20


def test_compute_var_hand_calculable():
    """
    For prices [100.0, 110.0, 88.0, 96.8] with sorted returns [-0.20, 0.10, 0.10]:
    Worst historical return is -20%, so VaR is bounded by 0.20.
    """
    prices = [100.0, 110.0, 88.0, 96.8]
    var_95 = compute_var(prices, confidence=0.95)
    assert 0.0 <= var_95 <= 0.20
    assert var_95 > 0.05  # should reflect the severe -20% drop


def test_compute_all_risk_metrics_wrapper():
    """Verify aggregated snapshot model output."""
    prices = [100.0, 105.0, 95.0, 102.0, 98.0]
    snapshot = compute_all_risk_metrics(prices)
    assert snapshot.sample_bars == 5
    assert snapshot.volatility_annualized > 0
    assert snapshot.max_drawdown < 0
    assert snapshot.var_95_daily >= 0


def test_edge_cases_empty_and_single_price():
    """Verify handling of edge case short histories."""
    assert compute_volatility([]) == 0.0
    assert compute_volatility([100.0]) == 0.0
    assert compute_max_drawdown([100.0]) == 0.0
    assert compute_var([]) == 0.0
