"""
graph module for StockSage multi-agent orchestration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.stocksage_graph import arun_stocksage, build_sequential_graph, run_stocksage

from src.graph.state import StockSageState, create_initial_state


def __getattr__(name: str):
    if name in ("build_sequential_graph", "run_stocksage", "arun_stocksage"):
        from src.graph.stocksage_graph import arun_stocksage, build_sequential_graph, run_stocksage
        mapping = {
            "build_sequential_graph": build_sequential_graph,
            "run_stocksage": run_stocksage,
            "arun_stocksage": arun_stocksage,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StockSageState",
    "arun_stocksage",
    "build_sequential_graph",
    "create_initial_state",
    "run_stocksage",
]
