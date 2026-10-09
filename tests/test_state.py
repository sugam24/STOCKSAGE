"""
tests/test_state.py
--------------------
Unit tests for Day 34 StockSageState schema and reducers.
"""

from src.graph.state import StockSageState, append_error, create_initial_state


def test_create_initial_state():
    """Verify initial state construction and normalization."""
    state = create_initial_state("nvda")
    assert state["ticker"] == "NVDA"
    assert state["stock_data"] == {}
    assert state["news_context"] == []
    assert state["risk_metrics"] == {}
    assert state["final_report"] == ""
    assert state["error_log"] == []


def test_append_error_reducer():
    """Verify append_error reducer properly combines error lists."""
    initial = ["DataAgent: network timeout"]
    update = ["NewsAgent: rate limit 429"]
    merged = append_error(initial, update)
    assert merged == [
        "DataAgent: network timeout",
        "NewsAgent: rate limit 429",
    ]

    # Test with empty or None values
    assert append_error(None, ["new error"]) == ["new error"]
    assert append_error(["existing"], None) == ["existing"]


def test_state_typeddict_compatibility():
    """Verify StockSageState accommodates partial updates."""
    state: StockSageState = {
        "ticker": "AAPL",
        "stock_data": {"price": 180.5},
    }
    assert state["ticker"] == "AAPL"
    assert state["stock_data"]["price"] == 180.5
