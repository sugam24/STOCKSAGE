"""
core/structured.py
------------------
Day 5: Structured output extraction via Gemini function-calling.

Steps covered:
  24. Convert a Pydantic model to the JSON-schema format Gemini expects
      using .model_json_schema().
  25. Send a request to Gemini with function calling enabled so the model
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
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Module-level Gemini client (initialised once; reused across calls)
# ---------------------------------------------------------------------------
_client: genai.Client | None = None

MODEL = "gemini-2.0-flash"


def _get_client() -> genai.Client:
    """Return (or lazily create) the shared Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Step 24 helper: Pydantic model → Gemini-compatible function declaration
# ---------------------------------------------------------------------------
def _make_function_declaration(schema: type[BaseModel]) -> types.FunctionDeclaration:
    """
    Convert a Pydantic model class into a Gemini FunctionDeclaration.

    Gemini's function-calling API expects each "tool" to be described as a
    function with a name, an optional description, and a JSON-schema for its
    parameters.  Pydantic's .model_json_schema() produces a JSON-schema dict
    that maps almost 1-to-1 onto what Gemini needs.

    Args:
        schema: A Pydantic BaseModel subclass (the class itself, not an instance).

    Returns:
        A FunctionDeclaration ready to be passed to Gemini's tools list.
    """
    # Step 24: use the built-in Pydantic helper
    json_schema: dict = schema.model_json_schema()

    # Gemini expects the parameter schema under the "parameters" key of the
    # FunctionDeclaration; the top-level keys (title, description, properties,
    # required) map directly.
    return types.FunctionDeclaration(
        name=json_schema.get("title", schema.__name__),
        description=(
            json_schema.get("description")
            or f"Extract {schema.__name__} data from the provided text."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                field: _convert_property(details)
                for field, details in json_schema.get("properties", {}).items()
            },
            "required": json_schema.get("required", []),
        },
    )


def _convert_property(prop: dict) -> dict:
    """
    Translate a single Pydantic JSON-schema property dict into the simplified
    Gemini parameter property format (which only uses type + description).
    """
    # Pydantic uses lowercase JSON Schema types ("string", "number", "integer");
    # Gemini wants uppercase OpenAPI-style types ("STRING", "NUMBER", "INTEGER").
    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    raw_type = prop.get("type", "string")
    gemini_type = type_map.get(raw_type, "STRING")

    result: dict = {"type": gemini_type}
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
      1. Converts *schema* into a Gemini FunctionDeclaration (Step 24).
      2. Sends *prompt* to Gemini with ``tool_choice="any"`` so the model is
         *forced* to call our function instead of returning free text (Step 25).
      3. Retrieves the function-call arguments from the response (Step 26).
      4. Validates those arguments into a real *schema* instance via
         ``schema.model_validate()`` (Step 26).

    Args:
        prompt: Natural-language text to extract information from.
        schema: A Pydantic BaseModel subclass describing the target structure.

    Returns:
        A validated instance of *schema*.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is missing.
        ValueError: If Gemini did not return a function call in its response.
        pydantic.ValidationError: If the returned arguments don't satisfy the schema.
    """
    client = _get_client()

    # Step 24 – build the function declaration from the Pydantic schema
    func_decl = _make_function_declaration(schema)
    tool = types.Tool(function_declarations=[func_decl])

    # Step 25 – send the request with function calling enabled
    # tool_config with mode ANY forces Gemini to always call a function.
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",  # force a function call every time
                )
            ),
            temperature=0.0,  # deterministic – we want exact extractions
        ),
    )

    # Step 26 – parse the function-call response
    # The response may contain multiple candidates; we inspect the first one.
    candidate = response.candidates[0]

    for part in candidate.content.parts:
        if part.function_call is not None:
            raw_args: dict = dict(part.function_call.args)
            # Step 26 – validate the raw dict into a real Pydantic object
            return schema.model_validate(raw_args)

    raise ValueError(
        "Gemini did not return a function call in its response.\n"
        f"Raw response text: {response.text!r}"
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
