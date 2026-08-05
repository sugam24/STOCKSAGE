"""
core/tools.py
-------------
Day 7: A generic ToolDispatcher + agentic while-loop (Groq-backed).

Steps covered:
  36. ToolDispatcher: holds a dict of {tool_name: callable} and builds the
      Groq tool list automatically from registered schemas.
  37. `run_agent()`: a while-loop that sends messages to Groq, checks
      whether the response contains a function call, executes the matching
      tool, appends the result, and loops – stopping only when the model
      returns plain text (no more tool calls).
  38. Two tools registered: get_current_price and get_pe_ratio.
      A test question that requires BOTH is answered end-to-end.
  39. Safety cap: the loop aborts after MAX_ITERATIONS (default 5) turns
      so a runaway tool chain can never hang the process.

Usage:
    uv run python -m src.core.tools
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"

MAX_ITERATIONS = 5          # Step 39 – safety cap


# ---------------------------------------------------------------------------
# Module-level client
# ---------------------------------------------------------------------------
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Schema normalisation helper
# ---------------------------------------------------------------------------

def _normalize_schema(obj: Any) -> Any:
    """Recursively lowercase the values of 'type' keys for OpenAI/Groq format."""
    if isinstance(obj, dict):
        return {
            k: (v.lower() if k == "type" and isinstance(v, str) else _normalize_schema(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_schema(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Step 36 – ToolDispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Registry that maps tool names to Python callables.

    Workflow
    --------
    1. Call `register(fn, schema)` for every tool you want to expose.
       *schema* is an OpenAPI-style dict with keys ``name``, ``description``,
       and ``parameters``.
    2. Read `.tools` to get the list of tool dicts to pass to Groq's
       ``tools=`` parameter.
    3. Call `dispatch(function_name, args_dict)` to execute a tool by name
       and get back a JSON-serialisable result.
    """

    def __init__(self) -> None:
        # Maps tool_name → Python callable
        self._registry: dict[str, Callable[..., Any]] = {}
        # Raw schema dicts – used to build tool format
        self._schemas: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, fn: Callable[..., Any], schema: dict) -> None:
        """
        Register *fn* under the name given in *schema["name"]*.

        Args:
            fn:     The Python callable to invoke when the model requests this tool.
            schema: OpenAPI-style dict with keys ``name``, ``description``,
                    and ``parameters``.

        Raises:
            KeyError:  If ``schema`` is missing the required ``"name"`` key.
            ValueError: If a tool with the same name has already been registered.
        """
        name: str = schema["name"]
        if name in self._registry:
            raise ValueError(
                f"A tool named '{name}' is already registered. "
                "Use a unique name per tool."
            )
        self._registry[name] = fn
        self._schemas.append(schema)

    @property
    def tools(self) -> list[dict]:
        """Return tools in OpenAI/Groq-compatible format (lowercase types)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": _normalize_schema(s["parameters"]),
                },
            }
            for s in self._schemas
        ]

    def dispatch(self, function_name: str, args: dict) -> Any:
        """
        Look up *function_name* in the registry and call it with *args*.

        Args:
            function_name: Exact name the model returned in the function_call.
            args:          Dict of keyword arguments extracted from the response.

        Returns:
            Whatever the registered callable returns.

        Raises:
            KeyError: If *function_name* is not in the registry.
        """
        if function_name not in self._registry:
            raise KeyError(
                f"Tool '{function_name}' is not registered. "
                f"Available tools: {list(self._registry)}"
            )
        return self._registry[function_name](**args)

    def __repr__(self) -> str:
        return f"ToolDispatcher(tools={list(self._registry)})"


# ---------------------------------------------------------------------------
# Step 37 – Agentic while-loop
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    dispatcher: ToolDispatcher,
    *,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> str:
    """
    Send *question* to Groq and resolve any tool calls automatically.

    The loop:
      1. Send the current conversation history to Groq.
      2. If the response contains tool_calls → dispatch each, append results,
         then loop again.
      3. If the response is plain text (no tool calls) → return it.
      4. If we reach *max_iterations* without a text answer → raise
         RuntimeError (Step 39 safety cap).

    Args:
        question:       Natural-language user question.
        dispatcher:     Populated ToolDispatcher with all needed tools.
        max_iterations: Maximum tool-call rounds before aborting (default 5).
        verbose:        Print each step to stdout when True.

    Returns:
        The model's final natural-language answer.

    Raises:
        RuntimeError: If the safety cap is reached without a final text answer.
    """
    client = _get_client()

    # Conversation history – grows as tool calls are resolved
    messages: list[dict] = [{"role": "user", "content": question}]
    tools = dispatcher.tools

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  User  → {question}")

    # Step 37 + Step 39 – bounded agentic loop
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"  [iteration {iteration}/{max_iterations}]")

        response = client.chat.completions.create(  # type: ignore[arg-type]
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message

        # ── No tool calls → model is done, return the text answer ─────────
        if not msg.tool_calls:
            final_text: str = msg.content or ""
            if verbose:
                print(f"  Model → {final_text.strip()}")
                print(f"{'─' * 60}\n")
            return final_text

        # ── Append the model turn (with tool_calls) ───────────────────────
        messages.append(msg)

        # ── Execute each tool call and collect results ────────────────────
        for tc in msg.tool_calls:
            name = tc.function.name
            args: dict = json.loads(tc.function.arguments)

            if verbose:
                print(f"  Model → function_call: {name}({args})")

            # Step 37 – dispatch to the registered Python function
            result = dispatcher.dispatch(name, args)

            if verbose:
                print(f"  Tool  → {name} returned: {result!r}")

            # Groq requires one "tool" role message per tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # Step 39 – safety cap: loop exhausted without a text response
    raise RuntimeError(
        f"Agent did not produce a final text answer within "
        f"{max_iterations} iterations."
    )


# ---------------------------------------------------------------------------
# Step 38 – Tool implementations
# ---------------------------------------------------------------------------

def get_current_price(ticker: str) -> float:
    """
    Fetch the latest market price for *ticker* from Yahoo Finance.

    Falls back to the previous close if the live price is unavailable
    (e.g. outside trading hours or for certain tickers).

    Args:
        ticker: Stock symbol in any case; normalised to uppercase internally.

    Returns:
        Price as a float (USD for US-listed stocks).

    Raises:
        ValueError: If no price data can be found for *ticker*.
    """
    info = yf.Ticker(ticker.upper()).fast_info
    price: float | None = getattr(info, "last_price", None)
    if price is None or (isinstance(price, float) and price != price):  # NaN
        price = getattr(info, "previous_close", None)
    if price is None:
        raise ValueError(f"Could not fetch a price for ticker '{ticker}'.")
    return float(price)


GET_CURRENT_PRICE_SCHEMA: dict = {
    "name": "get_current_price",
    "description": (
        "Return the current market price (in USD) for a publicly traded stock. "
        "Call this whenever the user asks about a stock's current or live price."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "ticker": {
                "type": "STRING",
                "description": (
                    "The stock ticker symbol to look up, e.g. 'AAPL', 'MSFT', 'TSLA'."
                ),
            }
        },
        "required": ["ticker"],
    },
}


def get_pe_ratio(ticker: str) -> float | None:
    """
    Fetch the trailing twelve-month P/E ratio for *ticker* from Yahoo Finance.

    Args:
        ticker: Stock symbol in any case; normalised to uppercase internally.

    Returns:
        P/E ratio as a float, or None if Yahoo Finance does not publish one
        (e.g. for companies with negative earnings).
    """
    info = yf.Ticker(ticker.upper()).info
    pe: float | None = info.get("trailingPE") or info.get("forwardPE")
    return float(pe) if pe is not None else None


GET_PE_RATIO_SCHEMA: dict = {
    "name": "get_pe_ratio",
    "description": (
        "Return the trailing twelve-month P/E ratio for a publicly traded stock. "
        "Call this when the user asks about valuation, P/E ratio, or whether a "
        "stock looks cheap or expensive relative to earnings."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "ticker": {
                "type": "STRING",
                "description": (
                    "The stock ticker symbol to look up, e.g. 'AAPL', 'MSFT', 'NVDA'."
                ),
            }
        },
        "required": ["ticker"],
    },
}


# ---------------------------------------------------------------------------
# Helper: build a pre-configured dispatcher with both tools registered
# ---------------------------------------------------------------------------

def build_default_dispatcher() -> ToolDispatcher:
    """
    Convenience factory: create and return a ToolDispatcher with
    get_current_price and get_pe_ratio already registered.
    """
    d = ToolDispatcher()
    d.register(get_current_price, GET_CURRENT_PRICE_SCHEMA)
    d.register(get_pe_ratio, GET_PE_RATIO_SCHEMA)
    return d


# ---------------------------------------------------------------------------
# Smoke-test / demo entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  StockSage — Day 7: ToolDispatcher + Agentic Loop Demo")
    print("=" * 60)

    dispatcher = build_default_dispatcher()
    print(f"\n📋  Registered tools: {dispatcher}\n")

    # Step 38 – question that requires BOTH tools
    questions = [
        # Needs get_current_price AND get_pe_ratio for AAPL
        "What is Apple's current stock price and its P/E ratio?",
        # Needs both tools for a different ticker
        "Compare MSFT's live price and P/E ratio. Is it expensive?",
    ]

    for i, q in enumerate(questions):
        if i > 0:
            time.sleep(2)   # small pause between requests
        try:
            answer = run_agent(q, dispatcher)
            print(f"  ✅ Final answer:\n     {answer.strip()}\n")
        except RuntimeError as exc:
            # Step 39 – safety cap triggered
            print(f"  🛑 Safety cap reached: {exc}\n")
        except Exception as exc:
            print(f"  ❌ Unexpected error: {exc}\n")
