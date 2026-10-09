"""
graph/intro.py
---------------
Day 33: Intro to LangGraph — State, Nodes, and Edges.

Demonstrates the core mechanics of LangGraph:
  1. A shared state container (TypedDict).
  2. Pure node functions that accept state and return updates.
  3. Directed edges linking execution flow between nodes.

Usage:
    uv run python -m src.graph.intro
"""

from __future__ import annotations

from typing import TypedDict
from langgraph.graph import END, START, StateGraph


class SimpleCountState(TypedDict):
    """Minimal state schema tracking an integer count and an event history."""
    count: int
    log: list[str]


def increment_node(state: SimpleCountState) -> dict:
    """
    Step 160: Node 1 — adds 1 to the count in state.
    """
    new_count = state["count"] + 1
    new_log = list(state.get("log", [])) + [f"increment_node: {state['count']} -> {new_count}"]
    return {"count": new_count, "log": new_log}


def double_node(state: SimpleCountState) -> dict:
    """
    Step 161: Node 2 — doubles the count in state.
    """
    new_count = state["count"] * 2
    new_log = list(state.get("log", [])) + [f"double_node: {state['count']} -> {new_count}"]
    return {"count": new_count, "log": new_log}


def build_single_node_graph():
    """Build a 1-node graph (START -> increment -> END)."""
    builder = StateGraph(SimpleCountState)
    builder.add_node("increment", increment_node)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


def build_two_node_graph():
    """Build a 2-node graph (START -> increment -> double -> END)."""
    builder = StateGraph(SimpleCountState)
    builder.add_node("increment", increment_node)
    builder.add_node("double", double_node)

    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile()


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 33: LangGraph Fundamentals Demo")
    print("=" * 65)

    print("\n1. Running Single-Node Graph (initial count = 0):")
    g1 = build_single_node_graph()
    r1 = g1.invoke({"count": 0, "log": []})
    print(f"   Final State: count={r1['count']}, log={r1['log']}")

    print("\n2. Running Two-Node Graph (initial count = 5):")
    g2 = build_two_node_graph()
    r2 = g2.invoke({"count": 5, "log": []})
    print(f"   Final State: count={r2['count']}")
    for entry in r2["log"]:
        print(f"     • {entry}")
