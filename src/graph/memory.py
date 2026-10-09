"""
graph/memory.py
----------------
Day 46: Long-Term Memory (Across Sessions).

Persists user preferences, search history, and previously analyzed tickers
across program restarts using a dedicated SQLite memory database.

Usage:
    uv run python -m src.graph.memory
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

DEFAULT_MEMORY_DB = os.path.join("data", "user_memory.db")


class UserMemoryStore:
    """
    Step 209: Long-term persistent store across program runs.
    """

    def __init__(self, db_path: str = DEFAULT_MEMORY_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables for preferences and ticker analysis history."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    query TEXT,
                    summary_snippet TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()

    def set_preference(self, key: str, value: Any) -> None:
        """Save a user preference (e.g. 'risk_tolerance', 'include_sec_filings')."""
        val_str = json.dumps(value)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, val_str, time.time()))
            conn.commit()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a saved user preference."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row["value"])
            return default

    def record_analysis(self, ticker: str, query: str = "", summary_snippet: str = "") -> None:
        """Record an analyzed stock ticker and timestamp to history."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO ticker_history (ticker, query, summary_snippet, timestamp)
                VALUES (?, ?, ?, ?)
            """, (ticker.strip().upper(), query, summary_snippet[:300], time.time()))
            conn.commit()

    def get_past_tickers(self, limit: int = 5) -> list[str]:
        """Return unique tickers analyzed recently in reverse chronological order."""
        with self._get_connection() as conn:
            cur = conn.execute("""
                SELECT ticker, MAX(timestamp) as last_ts
                FROM ticker_history
                GROUP BY ticker
                ORDER BY last_ts DESC
                LIMIT ?
            """, (limit,))
            return [row["ticker"] for row in cur.fetchall()]

    def load_user_context(self, current_ticker: str = "") -> dict[str, Any]:
        """
        Step 210: Load user context (preferences + recent tickers) for injection into state.
        """
        past_tickers = self.get_past_tickers(limit=5)
        risk_pref = self.get_preference("risk_tolerance", default="balanced")
        include_sec = self.get_preference("include_sec_filings", default=True)

        return {
            "past_tickers_analyzed": past_tickers,
            "risk_tolerance_preference": risk_pref,
            "include_sec_filings_preference": include_sec,
            "is_previously_analyzed": current_ticker.upper() in past_tickers if current_ticker else False,
        }


# Global convenience instance
_memory_store: UserMemoryStore | None = None


def get_memory_store(db_path: str = DEFAULT_MEMORY_DB) -> UserMemoryStore:
    global _memory_store
    if _memory_store is None or _memory_store.db_path != db_path:
        _memory_store = UserMemoryStore(db_path)
    return _memory_store


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 46: Long-Term Memory Demo")
    print("=" * 65)

    store = get_memory_store()
    store.set_preference("risk_tolerance", "conservative")
    store.record_analysis("NVDA", query="Analyze NVDA", summary_snippet="AI chip dominance")
    store.record_analysis("AAPL", query="What is AAPL price?", summary_snippet="Consumer ecosystem")

    context = store.load_user_context("NVDA")
    print("\nLoaded Cross-Session User Context:")
    for k, v in context.items():
        print(f"  • {k}: {v}")
