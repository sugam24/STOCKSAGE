"""
tests/test_technical.py
-----------------------
Tests for ingestion.technical — Steps 53–56.

Unit tests use a small fixed price list where RSI (period=3) is calculated
by hand.  Integration tests hit the Yahoo Finance API.

Run unit tests only:
    uv run pytest tests/test_technical.py -v -m "not integration"
"""

import pytest

from ingestion.technical import (
    TechnicalSignals,
    compute_moving_averages,
    compute_rsi,
    compute_technical_signals,
    get_technical_signals,
)


# ---------------------------------------------------------------------------
# Hand-calculated RSI fixture (period = 3)
# ---------------------------------------------------------------------------
#
# Prices:  [10, 12, 11, 13, 14, 12, 15]
# Changes: [ 2, -1,  2,  1, -2,  3]
# Gains:   [ 2,  0,  2,  1,  0,  3]
# Losses:  [ 0,  1,  0,  0,  2,  0]
#
# Index 3 (price=13): avg_gain=(2+0+2)/3=4/3, avg_loss=(0+1+0)/3=1/3
#   RS = 4  →  RSI = 100 − 100/5 = 80.0
#
# Index 4 (price=14): avg_gain=(4/3×2+1)/3=11/9, avg_loss=(1/3×2+0)/3=2/9
#   RS = 5.5  →  RSI = 100 − 100/6.5 ≈ 84.6154
#
# Index 5 (price=12): avg_gain=22/27, avg_loss=22/27
#   RS = 1  →  RSI = 50.0
#
# Index 6 (price=15): avg_gain=125/81, avg_loss=44/81
#   RS = 125/44  →  RSI ≈ 73.9645

HAND_CALC_PRICES = [10, 12, 11, 13, 14, 12, 15]
HAND_CALC_PERIOD = 3

# Expected RSI at each index (NaN for indices 0–2)
HAND_CALC_RSI = {
    3: 80.0,
    4: 84.61538461538461,
    5: 50.0,
    6: 73.96449704142012,
}


class TestComputeRsi:
    """Step 56 – verify RSI against hand-calculated values."""

    def test_hand_calculated_rsi_period_3(self):
        rsi = compute_rsi(HAND_CALC_PRICES, period=HAND_CALC_PERIOD)

        # First `period` values must be NaN (not enough history)
        for i in range(HAND_CALC_PERIOD):
            assert rsi.iloc[i] != rsi.iloc[i]  # NaN check

        for idx, expected in HAND_CALC_RSI.items():
            assert rsi.iloc[idx] == pytest.approx(expected, rel=1e-9), (
                f"RSI at index {idx}: got {rsi.iloc[idx]}, expected {expected}"
            )

    def test_rsi_default_period_requires_enough_data(self):
        rsi = compute_rsi(HAND_CALC_PRICES, period=14)
        assert rsi.isna().all()

    def test_rsi_overbought_and_oversold_bounds(self):
        """Monotonically rising prices → RSI should approach 100."""
        rising = list(range(1, 30))
        rsi = compute_rsi(rising, period=5)
        assert rsi.iloc[-1] == pytest.approx(100.0)

        falling = list(range(30, 1, -1))
        rsi = compute_rsi(falling, period=5)
        assert rsi.iloc[-1] == pytest.approx(0.0)


class TestComputeMovingAverages:
    """Step 54 – moving average helper."""

    def test_insufficient_data_returns_none(self):
        prices = list(range(1, 11))  # only 10 points
        ma20, ma50 = compute_moving_averages(prices)
        assert ma20 is None
        assert ma50 is None

    def test_ma20_with_enough_data(self):
        prices = list(range(1, 21))  # 1..20
        ma20, ma50 = compute_moving_averages(prices)
        assert ma20 == pytest.approx(10.5)  # mean of 1..20
        assert ma50 is None

    def test_ma50_with_enough_data(self):
        prices = list(range(1, 51))  # 1..50
        ma20, ma50 = compute_moving_averages(prices)
        assert ma20 == pytest.approx(40.5)  # mean of 31..50
        assert ma50 == pytest.approx(25.5)  # mean of 1..50


class TestTechnicalSignalsModel:
    """Step 55 – Pydantic model validation."""

    def test_minimal_valid(self):
        signals = TechnicalSignals(ticker="AAPL")
        assert signals.ticker == "AAPL"
        assert signals.rsi is None

    def test_round_trip(self):
        original = TechnicalSignals(
            ticker="AAPL",
            latest_close=311.0,
            rsi=55.3,
            ma20=323.72,
            ma50=310.5,
        )
        rebuilt = TechnicalSignals.model_validate(original.model_dump())
        assert rebuilt == original


class TestComputeTechnicalSignals:
    """Step 55 – combined function."""

    def test_combines_rsi_and_mas(self):
        signals = compute_technical_signals(
            HAND_CALC_PRICES,
            ticker="TEST",
            rsi_period=HAND_CALC_PERIOD,
        )
        assert isinstance(signals, TechnicalSignals)
        assert signals.ticker == "TEST"
        assert signals.latest_close == 15.0
        assert signals.rsi == pytest.approx(73.9645, rel=1e-3)
        assert signals.ma20 is None  # only 7 data points
        assert signals.ma50 is None


@pytest.mark.integration
class TestGetTechnicalSignals:
    """Integration test – live Yahoo Finance API."""

    def test_aapl_returns_model(self):
        signals = get_technical_signals("AAPL", period="6mo")
        assert isinstance(signals, TechnicalSignals)
        assert signals.ticker == "AAPL"
        assert signals.latest_close is not None
        assert signals.latest_close > 0
        assert signals.rsi is not None
        assert 0 <= signals.rsi <= 100
        assert signals.ma20 is not None
