"""
core/structured.py
------------------
Day 5: Structured output extraction via Groq function-calling.

Steps covered:
  24. Convert a Pydantic model to the JSON-schema format Groq expects
      using .model_json_schema().
  25. Send a request to Groq with function calling enabled so the model
      is *forced* to return structured data matching the schema.
  26. Parse the function-call response and validate it into a real
      Pydantic object.
  27. Expose a single reusable public API:
          extract(prompt: str, schema: type[BaseModel]) -> BaseModel

Usage (quick smoke-test):
    uv run python -m src.core.structured
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Module-level Groq client (initialised once; reused across calls)
# ---------------------------------------------------------------------------
_client: Groq | None = None

MODEL = "llama-3.3-70b-versatile"


def _get_client() -> Groq:
    """Return (or lazily create) the shared Groq client."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Step 24 helper: Pydantic model → Groq-compatible tool declaration
# ---------------------------------------------------------------------------
def _make_function_declaration(schema: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model class into a Groq-compatible tool declaration.

    Groq's function-calling API (OpenAI-compatible) expects each "tool" to
    be described with a name, description, and a standard JSON Schema for
    its parameters.  Pydantic's .model_json_schema() produces exactly this.

    Args:
        schema: A Pydantic BaseModel subclass (the class itself, not an instance).

    Returns:
        A dict in OpenAI/Groq tool format ready to pass to the tools list.
    """
    # Step 24: use the built-in Pydantic helper
    json_schema: dict = schema.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": json_schema.get("title", schema.__name__),
            "description": (
                json_schema.get("description")
                or f"Extract {schema.__name__} data from the provided text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    field: _convert_property(details)
                    for field, details in json_schema.get("properties", {}).items()
                },
                "required": json_schema.get("required", []),
            },
        },
    }


def _convert_property(prop: dict) -> dict:
    """
    Translate a single Pydantic JSON-schema property dict into the simplified
    parameter property format (type + description).

    Groq/OpenAI uses standard lowercase JSON Schema types, so no
    case conversion is needed (Pydantic already outputs lowercase).
    """
    result: dict = {"type": prop.get("type", "string")}
    if "description" in prop:
        result["description"] = prop["description"]
    return result


# ---------------------------------------------------------------------------
# Step 25 / 26 / 27: The public extract() function
# ---------------------------------------------------------------------------
def extract(prompt: str, schema: type[T]) -> T:
    """
    Extract structured data from *prompt* and validate it into a *schema* object.

    This function:
      1. Converts *schema* into a Groq tool declaration (Step 24).
      2. Sends *prompt* to Groq with ``tool_choice`` forcing a specific
         function call so the model always returns structured data (Step 25).
      3. Retrieves the function-call arguments from the response (Step 26).
      4. Validates those arguments into a real *schema* instance via
         ``schema.model_validate()`` (Step 26).

    Args:
        prompt: Natural-language text to extract information from.
        schema: A Pydantic BaseModel subclass describing the target structure.

    Returns:
        A validated instance of *schema*.

    Raises:
        EnvironmentError: If GROQ_API_KEY is missing.
        ValueError: If Groq did not return a function call in its response.
        pydantic.ValidationError: If the returned arguments don't satisfy the schema.
    """
    client = _get_client()

    # Step 24 – build the tool declaration from the Pydantic schema
    tool_decl = _make_function_declaration(schema)
    func_name = tool_decl["function"]["name"]

    # Step 25 – send the request with function calling enabled
    # tool_choice with a specific function forces Groq to always call it.
    response = client.chat.completions.create(  # type: ignore[arg-type]
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool_decl],
        tool_choice={
            "type": "function",
            "function": {"name": func_name},
        },
        temperature=0.0,  # deterministic – we want exact extractions
    )

    # Step 26 – parse the function-call response
    msg = response.choices[0].message

    if msg.tool_calls:
        raw_args: dict = json.loads(msg.tool_calls[0].function.arguments)
        # Step 26 – validate the raw dict into a real Pydantic object
        return schema.model_validate(raw_args)

    raise ValueError(
        "Groq did not return a function call in its response.\n"
        f"Raw response text: {msg.content!r}"
    )


# ---------------------------------------------------------------------------
# Step 28: Quick smoke-test – run with:  uv run python -m src.core.structured
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .schema import StockData

    examples = [
        # Paragraph 1 – straightforward mention
        (
            "Apple Inc. (AAPL) closed at $189.30 today. "
            "Analysts note that its price-to-earnings ratio sits at 28.5, "
            "reflecting strong investor confidence in its services segment."
        ),
        # Paragraph 2 – narrative style
        (
            "Microsoft shares (MSFT) have been trading around the $415 mark. "
            "The tech giant's PE ratio has risen to approximately 34.2 following "
            "its AI-driven cloud revenue beat last quarter."
        ),
        # Paragraph 3 – brief / telegraphic
        (
            "NVDA: price $875.40, P/E 68.3. "
            "GPU demand continues to outpace supply amid the generative-AI buildout."
        ),
    ]

    print("=" * 60)
    print("  StockSage — Day 5: Structured Extraction Demo")
    print("=" * 60)

    for i, paragraph in enumerate(examples, start=1):
        print(f"\n📄 Paragraph {i}:")
        print(f"   {paragraph}\n")
        try:
            stock: StockData = extract(paragraph, StockData)
            print(f"   ✅ Extracted → {stock}")
            print(f"   📦 model_dump → {stock.model_dump()}")
        except Exception as exc:
            print(f"   ❌ Error: {exc}")
        print("-" * 60)
