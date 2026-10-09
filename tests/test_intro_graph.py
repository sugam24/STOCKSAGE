"""
tests/test_intro_graph.py
--------------------------
Unit tests for Day 33 LangGraph intro graphs.
"""

from src.graph.intro import (
    build_single_node_graph,
    build_two_node_graph,
    double_node,
    increment_node,
)


def test_individual_nodes():
    """Verify nodes as standalone pure functions."""
    inc_res = increment_node({"count": 10, "log": []})
    assert inc_res["count"] == 11
    assert len(inc_res["log"]) == 1

    dbl_res = double_node({"count": 11, "log": []})
    assert dbl_res["count"] == 22
    assert len(dbl_res["log"]) == 1


def test_single_node_graph_execution():
    """Verify single-node compiled graph execution."""
    graph = build_single_node_graph()
    result = graph.invoke({"count": 41, "log": []})
    assert result["count"] == 42
    assert len(result["log"]) == 1


def test_two_node_graph_execution():
    """Verify two-node compiled graph transitions across edges."""
    graph = build_two_node_graph()
    result = graph.invoke({"count": 3, "log": []})
    # (3 + 1) * 2 = 8
    assert result["count"] == 8
    assert len(result["log"]) == 2
    assert "increment_node" in result["log"][0]
    assert "double_node" in result["log"][1]
