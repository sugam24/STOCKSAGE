"""
core/tool_calling.py
--------------------
Day 6: Gemini Function Calling with a real live tool.

Steps covered:
  30. get_current_price(ticker) – plain Python function using yfinance.
  31. JSON schema description written by hand (PRICE_TOOL_SCHEMA / FunctionDeclaration).
  32. Send Gemini a message with the tool available and ask about AAPL.
  33. Detect a function-call response, run the real function, send result back.
  34. Print Gemini's final natural-language answer with the live price.

Usage:
    uv run python -m src.core.tool_calling

Note on quota:
    The free-tier Gemini API allows ~15 RPM / limited daily tokens.
    This module uses automatic exponential-backoff retry (up to 3 attempts)
    on 429 RESOURCE_EXHAUSTED errors so it recovers gracefully from short
    rate-limit windows.
"""

from __future__ import annotations

import json
import os
import time

import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Primary model; falls back automatically on 429.
MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-2.0-flash-lite"

# Retry settings for 429 / quota errors
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 5  # seconds (doubles each attempt)

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
# Step 31 – JSON schema written by hand
# ---------------------------------------------------------------------------
# This is what Gemini actually sees: a FunctionDeclaration whose `parameters`
# field is an OpenAPI-style JSON Schema object.
#
# In raw dict form (so you can read exactly what travels over the wire):
PRICE_TOOL_SCHEMA: dict = {
    "name": "get_current_price",
    "description": (
        "Return the current market price for a publicly traded stock. "
        "Call this whenever the user asks about a stock's current or live price."
    ),
    "parameters": {
        "type": "OBJECT",           # <-- Gemini uses uppercase OpenAPI types
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

# Wrap the raw dict in a proper SDK object (Gemini SDK accepts both, but the
# typed object gives us IDE autocompletion and runtime validation).
_price_function_decl = types.FunctionDeclaration(
    name=PRICE_TOOL_SCHEMA["name"],
    description=PRICE_TOOL_SCHEMA["description"],
    parameters=PRICE_TOOL_SCHEMA["parameters"],
)

_price_tool = types.Tool(function_declarations=[_price_function_decl])


# ---------------------------------------------------------------------------
# Module-level Gemini client
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """
    Wrapper around client.models.generate_content that retries on 429 errors
    with exponential backoff, transparently falling back to FALLBACK_MODEL
    if the primary model's daily quota is exhausted.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            msg = str(exc)
            is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if is_quota and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  ⚠️  Rate-limited (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Retrying in {delay}s…"
                )
                time.sleep(delay)
            elif is_quota and model != FALLBACK_MODEL:
                # Try the fallback model for the last attempt
                print(
                    f"  ⚠️  Quota exhausted on {model}. "
                    f"Switching to {FALLBACK_MODEL}…"
                )
                return client.models.generate_content(
                    model=FALLBACK_MODEL, contents=contents, config=config
                )
            else:
                raise


# ---------------------------------------------------------------------------
# Steps 32–34 – full round-trip
# ---------------------------------------------------------------------------

def ask_about_stock(question: str) -> str:
    """
    Ask Gemini a question that may require a live stock price lookup.

    The conversation follows this pattern:
      Turn 1 (user → model) : question + tool available
      Turn 2 (model → user) : function-call request          [Step 32 / 33]
      Turn 3 (user → model) : function result
      Turn 4 (model → user) : final natural-language answer  [Step 34]

    Args:
        question: Any natural-language question, e.g. "What's AAPL trading at?"

    Returns:
        Gemini's final answer as a string.
    """
    client = _get_client()

    # ── Turn 1: send the user's question with the tool declared ─────────────
    # Step 32 – tool is available but NOT forced (mode AUTO lets Gemini decide
    # whether to call it; it will always call it for a price question).
    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )
    ]

    print(f"\n{'─'*60}")
    print(f"  User  → {question}")

    response = _generate_with_retry(
        client=client,
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[_price_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            temperature=0.0,
        ),
    )

    # ── Turn 2: inspect the response ────────────────────────────────────────
    # Step 33 – detect whether Gemini issued a function-call
    candidate = response.candidates[0]
    function_call_part: types.Part | None = None

    for part in candidate.content.parts:
        if part.function_call is not None:
            function_call_part = part
            break

    # If Gemini answered directly (no tool needed), return the text as-is.
    if function_call_part is None:
        final_text = response.text
        print(f"  Model → (no tool call) {final_text}")
        return final_text

    # ── Tool execution (Step 33 cont.) ──────────────────────────────────────
    fc = function_call_part.function_call
    call_args: dict = dict(fc.args)
    ticker: str = call_args["ticker"].upper()

    print(f"  Model → function_call: {fc.name}({call_args})")

    # Run the REAL Python function
    live_price: float = get_current_price(ticker)
    print(f"  Tool  → {ticker} price = ${live_price:.2f}")

    # ── Turn 3: send the function result back to Gemini ──────────────────────
    # We must append both the model's function-call turn and our tool-result
    # turn to the conversation history before calling again.
    contents.append(
        types.Content(
            role="model",
            parts=[function_call_part],
        )
    )
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": live_price},
                    )
                )
            ],
        )
    )

    # ── Turn 4: get Gemini's final natural-language answer (Step 34) ─────────
    final_response = _generate_with_retry(
        client=client,
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[_price_tool],
            temperature=0.0,
        ),
    )

    final_text: str = final_response.text
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
            # Small pause between questions to respect per-minute rate limits.
            time.sleep(3)
        try:
            answer = ask_about_stock(q)
        except Exception as exc:
            print(f"  ❌ Error: {exc}\n")

    # Also print the raw schema so the reader can see exactly what Gemini sees.
    print("\n📋  Raw JSON Schema sent to Gemini (Step 31):")
    print(json.dumps(PRICE_TOOL_SCHEMA, indent=2))
