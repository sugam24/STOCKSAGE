"""
graph module for StockSage multi-agent orchestration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.memory import UserMemoryStore, get_memory_store
    from src.graph.portfolio import analyze_portfolio
    from src.graph.stocksage_graph import (
        arun_stocksage_turn,
        build_sequential_graph,
        build_stocksage_graph,
        get_sqlite_checkpointer,
        run_stocksage,
    )

from src.graph.state import StockSageState, create_initial_state


def __getattr__(name: str):
    if name in ("UserMemoryStore", "get_memory_store"):
        from src.graph import memory
        return getattr(memory, name)
    if name == "analyze_portfolio":
        from src.graph import portfolio
        return getattr(portfolio, name)

    targets = (
        "build_sequential_graph",
        "build_stocksage_graph",
        "run_stocksage",
        "arun_stocksage_turn",
        "get_sqlite_checkpointer",
    )
    if name in targets:
        from src.graph import stocksage_graph
        return getattr(stocksage_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StockSageState",
    "UserMemoryStore",
    "analyze_portfolio",
    "arun_stocksage_turn",
    "build_sequential_graph",
    "build_stocksage_graph",
    "create_initial_state",
    "get_memory_store",
    "get_sqlite_checkpointer",
    "run_stocksage",
]
