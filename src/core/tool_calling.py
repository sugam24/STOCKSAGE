"""
core/tool_calling.py
--------------------
Day 6: Groq Function Calling with a real live tool.

Steps covered:
  30. get_current_price(ticker) – plain Python function using yfinance.
  31. JSON schema description written by hand (PRICE_TOOL_SCHEMA).
  32. Send Groq a message with the tool available and ask about AAPL.
  33. Detect a function-call response, run the real function, send result back.
  34. Print Groq's final natural-language answer with the live price.

Usage:
    uv run python -m src.core.tool_calling
"""

from __future__ import annotations

import json
import os
import time

import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Step 30 – plain Python function
# ---------------------------------------------------------------------------

def get_current_price(ticker: str) -> float:
    """
    Fetch the latest market price for *ticker* from Yahoo Finance.

    Uses yfinance's fast_info shortcut, which is a single lightweight HTTP
    request (no full history download).  Falls back to the previous close
    if the live price is unavailable (e.g. outside trading hours).

    Args:
        ticker: Uppercase stock symbol, e.g. "AAPL" or "MSFT".

    Returns:
        The current (or most-recent) price as a float.

    Raises:
        ValueError: If no price data could be found for *ticker*.
    """
    info = yf.Ticker(ticker).fast_info

    # fast_info exposes .last_price (NaN outside market hours on some tickers)
    price: float | None = getattr(info, "last_price", None)

    # Fallback: previous close is always available
    if price is None or (isinstance(price, float) and price != price):  # NaN check
        price = getattr(info, "previous_close", None)

    if price is None:
        raise ValueError(f"Could not fetch a price for ticker '{ticker}'.")

    return float(price)


# ---------------------------------------------------------------------------
# Step 31 – JSON schema written by hand (OpenAI/Groq format)
# ---------------------------------------------------------------------------
# This is what Groq sees: a tool definition in OpenAI-compatible format.
PRICE_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_current_price",
        "description": (
            "Return the current market price for a publicly traded stock. "
            "Call this whenever the user asks about a stock's current or live price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "The stock ticker symbol to look up, e.g. 'AAPL', 'MSFT', 'TSLA'."
                    ),
                }
            },
            "required": ["ticker"],
        },
    },
}


# ---------------------------------------------------------------------------
# Module-level Groq client
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
# Steps 32–34 – full round-trip
# ---------------------------------------------------------------------------

def ask_about_stock(question: str) -> str:
    """
    Ask Groq a question that may require a live stock price lookup.

    The conversation follows this pattern:
      Turn 1 (user → model) : question + tool available
      Turn 2 (model → user) : function-call request          [Step 32 / 33]
      Turn 3 (user → model) : function result
      Turn 4 (model → user) : final natural-language answer  [Step 34]

    Args:
        question: Any natural-language question, e.g. "What's AAPL trading at?"

    Returns:
        Groq's final answer as a string.
    """
    client = _get_client()

    # ── Turn 1: send the user's question with the tool declared ─────────────
    # Step 32 – tool is available but NOT forced (tool_choice="auto" lets
    # the model decide whether to call it).
    messages: list[dict] = [{"role": "user", "content": question}]

    print(f"\n{'─'*60}")
    print(f"  User  → {question}")

    response = client.chat.completions.create(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,
        tools=[PRICE_TOOL_SCHEMA],
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message

    # ── Turn 2: inspect the response ────────────────────────────────────────
    # Step 33 – detect whether the model issued a function-call

    # If the model answered directly (no tool needed), return the text as-is.
    if not msg.tool_calls:
        final_text = msg.content or ""
        print(f"  Model → (no tool call) {final_text}")
        return final_text

    # ── Tool execution (Step 33 cont.) ──────────────────────────────────────
    tc = msg.tool_calls[0]
    call_args: dict = json.loads(tc.function.arguments)
    ticker: str = call_args["ticker"].upper()

    print(f"  Model → function_call: {tc.function.name}({call_args})")

    # Run the REAL Python function
    live_price: float = get_current_price(ticker)
    print(f"  Tool  → {ticker} price = ${live_price:.2f}")

    # ── Turn 3: send the function result back ────────────────────────────────
    # Append the model's tool-call turn and our tool-result turn to history.
    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": str(live_price),
    })

    # ── Turn 4: get the final natural-language answer (Step 34) ──────────────
    final_response = client.chat.completions.create(  # type: ignore[arg-type]
        model=MODEL,
        messages=messages,
        tools=[PRICE_TOOL_SCHEMA],
        temperature=0.0,
    )

    final_text: str = final_response.choices[0].message.content or ""
    print(f"  Model → {final_text.strip()}")
    print(f"{'─'*60}\n")
    return final_text


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  StockSage — Day 6: Function Calling Demo")
    print("=" * 60)

    # Step 32 – the canonical question from the task description
    questions = [
        "What's AAPL trading at?",
        "How much does a share of Microsoft cost right now?",
    ]

    for i, q in enumerate(questions):
        if i > 0:
            # Small pause between questions to be polite.
            time.sleep(2)
        try:
            answer = ask_about_stock(q)
        except Exception as exc:
            print(f"  ❌ Error: {exc}\n")

    # Also print the raw schema so the reader can see exactly what Groq sees.
    print("\n📋  Raw JSON Schema sent to Groq (Step 31):")
    print(json.dumps(PRICE_TOOL_SCHEMA, indent=2))
