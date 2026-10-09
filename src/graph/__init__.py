"""
graph module for StockSage multi-agent orchestration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.stocksage_graph import (
        arun_stocksage_turn,
        build_sequential_graph,
        build_stocksage_graph,
        get_sqlite_checkpointer,
        run_stocksage,
    )

from src.graph.state import StockSageState, create_initial_state


def __getattr__(name: str):
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
    "arun_stocksage_turn",
    "build_sequential_graph",
    "build_stocksage_graph",
    "create_initial_state",
    "get_sqlite_checkpointer",
    "run_stocksage",
]
