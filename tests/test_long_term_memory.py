"""
tests/test_long_term_memory.py
-------------------------------
Unit tests for Day 46 Long-Term Memory (Across Sessions).
"""

import os
import tempfile
from src.graph.memory import UserMemoryStore


def test_preferences_persistence():
    """Verify preferences are saved and loaded accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_user_memory.db")
        store1 = UserMemoryStore(db_path)

        store1.set_preference("risk_tolerance", "aggressive")
        store1.set_preference("max_portfolio_allocation", 0.25)
        assert store1.get_preference("risk_tolerance") == "aggressive"
        assert store1.get_preference("max_portfolio_allocation") == 0.25

        # Reopen in new instance (simulate program restart)
        store2 = UserMemoryStore(db_path)
        assert store2.get_preference("risk_tolerance") == "aggressive"
        assert store2.get_preference("max_portfolio_allocation") == 0.25


def test_ticker_history_and_context_loading():
    """Verify ticker history accumulation across sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_user_memory.db")
        store1 = UserMemoryStore(db_path)

        store1.record_analysis("AAPL", query="Analyze Apple", summary_snippet="iPhone strength")
        store1.record_analysis("NVDA", query="Analyze NVDA", summary_snippet="AI Blackwell chips")

        # Reopen in new instance (simulate program restart)
        store2 = UserMemoryStore(db_path)
        past_tickers = store2.get_past_tickers(limit=5)
        assert "NVDA" in past_tickers
        assert "AAPL" in past_tickers

        context = store2.load_user_context("NVDA")
        assert context["is_previously_analyzed"] is True
        assert context["is_previously_analyzed"] is True
        assert "NVDA" in context["past_tickers_analyzed"]

        # Check unanalyzed ticker
        context_tsla = store2.load_user_context("TSLA")
        assert context_tsla["is_previously_analyzed"] is False
