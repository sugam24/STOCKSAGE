"""
tests/test_tools.py
-------------------
Unit and integration tests for core/tools.py (Groq-backed).

Unit tests (no network / no Groq API calls):
  - ToolDispatcher.register() happy-path
  - ToolDispatcher.register() duplicate-name guard
  - ToolDispatcher.dispatch() happy-path
  - ToolDispatcher.dispatch() unknown-name guard
  - ToolDispatcher.tools returns correct OpenAI/Groq tool structure
  - run_agent() safety cap (Step 39): RuntimeError after MAX_ITERATIONS

Integration tests (live API calls – skipped unless --run-integration flag or
  STOCKSAGE_INTEGRATION env-var is set):
  - run_agent() with both tools answers a dual-tool question (Step 38)
  - Safety cap is NOT triggered on a normal single-tool question
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.tools import (
    GET_CURRENT_PRICE_SCHEMA,
    GET_PE_RATIO_SCHEMA,
    MAX_ITERATIONS,
    ToolDispatcher,
    build_default_dispatcher,
    get_current_price,
    get_pe_ratio,
    run_agent,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _add(a: float, b: float) -> float:
    return a + b


ADD_SCHEMA: dict = {
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "a": {"type": "NUMBER", "description": "First operand."},
            "b": {"type": "NUMBER", "description": "Second operand."},
        },
        "required": ["a", "b"],
    },
}


def _make_text_response(text: str):
    """
    Build a minimal mock Groq ChatCompletion response with plain text
    (no tool calls).
    """
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_function_call_response(name: str, args: dict):
    """
    Build a mock Groq response whose message contains a tool_call.
    """
    tc = MagicMock()
    tc.id = "call_test_123"
    tc.function.name = name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Step 36 – ToolDispatcher unit tests
# ---------------------------------------------------------------------------

class TestToolDispatcher:

    def test_register_and_dispatch_happy_path(self):
        """Registered function is callable through dispatch()."""
        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)
        result = d.dispatch("add", {"a": 3.0, "b": 4.0})
        assert result == 7.0

    def test_register_duplicate_raises(self):
        """Registering two tools with the same name raises ValueError."""
        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)
        with pytest.raises(ValueError, match="already registered"):
            d.register(_add, ADD_SCHEMA)

    def test_dispatch_unknown_name_raises(self):
        """Dispatching an unregistered tool name raises KeyError."""
        d = ToolDispatcher()
        with pytest.raises(KeyError, match="not registered"):
            d.dispatch("nonexistent", {})

    def test_tools_returns_openai_format(self):
        """tools property returns OpenAI/Groq-compatible tool dicts."""
        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)
        tools = d.tools
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "add"

    def test_tools_multiple_tools(self):
        """All registered tools appear in the tools list."""
        d = build_default_dispatcher()
        tools = d.tools
        names = {t["function"]["name"] for t in tools}
        assert "get_current_price" in names
        assert "get_pe_ratio" in names

    def test_repr_includes_tool_names(self):
        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)
        assert "add" in repr(d)

    def test_build_default_dispatcher_registers_both(self):
        """build_default_dispatcher registers exactly the two expected tools."""
        d = build_default_dispatcher()
        # Both schemas should be dispatchable (smoke dispatch is in integration)
        assert "get_current_price" in repr(d)
        assert "get_pe_ratio" in repr(d)


# ---------------------------------------------------------------------------
# Step 37 / 39 – run_agent() unit tests (mocked Groq)
# ---------------------------------------------------------------------------

class TestRunAgentMocked:
    """All Groq API calls are mocked; no network required."""

    @patch("src.core.tools._get_client")
    def test_plain_text_response_returns_immediately(self, mock_get_client):
        """If the model answers with plain text on turn 1, return it directly."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_text_response(
            "AAPL is trading at $200."
        )
        mock_get_client.return_value = mock_client

        d = ToolDispatcher()
        result = run_agent("What is AAPL price?", d, verbose=False)
        assert "AAPL" in result or "$200" in result
        assert mock_client.chat.completions.create.call_count == 1

    @patch("src.core.tools._get_client")
    def test_single_tool_call_resolved(self, mock_get_client):
        """One function call is dispatched and result fed back before final answer."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_function_call_response("add", {"a": 1.0, "b": 2.0}),
            _make_text_response("The answer is 3."),
        ]
        mock_get_client.return_value = mock_client

        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)

        result = run_agent("What is 1 + 2?", d, verbose=False)
        assert "3" in result
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.core.tools._get_client")
    def test_safety_cap_raises_runtime_error(self, mock_get_client):
        """
        Step 39 – if the model keeps returning function calls indefinitely,
        run_agent raises RuntimeError after max_iterations rounds.
        """
        mock_client = MagicMock()
        # Always return a function call – never a final text answer
        mock_client.chat.completions.create.return_value = _make_function_call_response(
            "add", {"a": 1.0, "b": 1.0}
        )
        mock_get_client.return_value = mock_client

        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)

        with pytest.raises(RuntimeError, match="iterations"):
            run_agent("Loop forever", d, max_iterations=3, verbose=False)

        # Should have called the API exactly max_iterations times
        assert mock_client.chat.completions.create.call_count == 3

    @patch("src.core.tools._get_client")
    def test_safety_cap_default_is_max_iterations_constant(self, mock_get_client):
        """The default max_iterations matches the module-level MAX_ITERATIONS."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_function_call_response(
            "add", {"a": 0.0, "b": 0.0}
        )
        mock_get_client.return_value = mock_client

        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)

        with pytest.raises(RuntimeError):
            run_agent("Loop", d, verbose=False)

        assert mock_client.chat.completions.create.call_count == MAX_ITERATIONS

    @patch("src.core.tools._get_client")
    def test_tool_result_appended_to_history(self, mock_get_client):
        """
        After a function call the tool result is passed back so the model can
        compose its final answer.  Verify the second API call receives a
        tool-role message in its messages.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_function_call_response("add", {"a": 10.0, "b": 5.0}),
            _make_text_response("Sum is 15."),
        ]
        mock_get_client.return_value = mock_client

        d = ToolDispatcher()
        d.register(_add, ADD_SCHEMA)

        run_agent("10 + 5?", d, verbose=False)

        # Inspect the `messages` list passed to the SECOND API call
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        second_messages: list = second_call_kwargs["messages"]

        # The last element should be a tool-role message with the result
        last_msg = second_messages[-1]
        assert last_msg["role"] == "tool"
        assert "15.0" in last_msg["content"]


# ---------------------------------------------------------------------------
# Step 38 – Integration tests (require live API + yfinance)
# ---------------------------------------------------------------------------

INTEGRATION = bool(os.getenv("STOCKSAGE_INTEGRATION"))


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION, reason="Set STOCKSAGE_INTEGRATION=1 to run")
class TestRunAgentIntegration:

    def test_dual_tool_question_aapl(self):
        """
        Step 38 – Ask a question that requires both get_current_price AND
        get_pe_ratio.  The model should call both tools and return a coherent answer.
        """
        d = build_default_dispatcher()
        answer = run_agent(
            "What is Apple's current stock price and its P/E ratio?",
            d,
            verbose=True,
        )
        # Minimal sanity: the answer should mention AAPL or Apple and a number
        lower = answer.lower()
        assert "apple" in lower or "aapl" in lower, f"Unexpected answer: {answer!r}"
        # Should contain at least one dollar sign or numeric value
        has_number = any(ch.isdigit() for ch in answer)
        assert has_number, f"No numeric content in answer: {answer!r}"

    def test_single_tool_does_not_trigger_safety_cap(self):
        """
        A simple price-only question should resolve in well under MAX_ITERATIONS.
        """
        d = build_default_dispatcher()
        # Should NOT raise RuntimeError
        answer = run_agent(
            "What is MSFT trading at right now?",
            d,
            max_iterations=MAX_ITERATIONS,
            verbose=True,
        )
        assert answer.strip(), "Expected a non-empty answer."

    def test_get_current_price_returns_positive_float(self):
        """Unit-level integration: get_current_price should return a positive price."""
        price = get_current_price("AAPL")
        assert isinstance(price, float)
        assert price > 0, f"Price should be positive, got {price}"

    def test_get_pe_ratio_returns_float_or_none(self):
        """Unit-level integration: get_pe_ratio should return float or None."""
        pe = get_pe_ratio("AAPL")
        assert pe is None or (isinstance(pe, float) and pe > 0), (
            f"Expected positive float or None, got {pe!r}"
        )
