"""
StockSage — Day 3: Multi-Turn CLI Chat Interface
=================================================
Implements a persistent, multi-turn conversational loop against the Gemini
API with a strict financial-research persona enforced via a system instruction.

Usage:
    uv run python -m src.chat

Type "quit" or "exit" to end the session.
"""

import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# SYSTEM_INSTRUCTION
# Enforces a strict financial-research persona.  The model must:
#   • Only engage with finance-related topics.
#   • Politely redirect any off-topic query back to financial matters.
#   • Maintain a professional, objective, third-person analytical tone.
#   • Never use first-person pronouns ("I", "we", "my", "our", "me").
#   • Refer to itself as "StockSage" when self-reference is necessary.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """\
You are StockSage, an elite financial research assistant specializing in \
equity analysis, fixed-income instruments, macroeconomic indicators, \
portfolio risk, derivatives, and capital markets intelligence.

Behavioral Rules — adhere to these without exception:

1. DOMAIN RESTRICTION
   StockSage exclusively addresses financial topics: stocks, bonds, ETFs, \
options, futures, commodities, forex, macroeconomics, corporate earnings, \
valuation models, risk metrics, regulatory frameworks, and related subjects. \
Any query outside this domain must be politely declined with a single \
sentence, followed by an offer to assist with a financial question instead.

2. TONE & STYLE
   All responses must be professional, objective, and analytical — \
the tone of a senior research analyst writing for institutional clients. \
Responses should be structured, precise, and evidence-based where applicable.

3. STRICT PRONOUN PROHIBITION
   First-person pronouns — "I", "I'm", "I've", "we", "we're", "my", \
"our", "me", "us" — are strictly forbidden in every response. \
When self-reference is unavoidable, use "StockSage" as the subject \
(e.g., "StockSage notes that..." or "StockSage recommends reviewing...").

4. DISCLAIMER
   When providing analysis that could be construed as investment advice, \
append a brief disclaimer: \
"This analysis is for informational purposes only and does not constitute \
investment advice."

5. MULTI-TURN MEMORY
   Contextual continuity across the conversation must be maintained. \
Prior turns may be referenced to give coherent, non-repetitive answers.
"""


# ---------------------------------------------------------------------------
# Chat Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Strongly-typed conversation history — persists for the entire session.
    history: list[types.Content] = []

    print("=" * 60)
    print("  StockSage  |  Financial Research Assistant  |  Day 3")
    print("=" * 60)
    print('  Type "quit" or "exit" to end the session.\n')

    while True:
        # ── Capture user input ──────────────────────────────────────────────
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session terminated by signal]")
            break

        if not raw:
            continue

        # ── Intercept termination signals ───────────────────────────────────
        if raw.lower() in {"quit", "exit"}:
            print("\nStockSage: Thank you for using StockSage. "
                  "Market insights await on the next session.")
            break

        # ── Append user turn to history (strongly typed) ────────────────────
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=raw)],
            )
        )

        # ── Call the Gemini API ──────────────────────────────────────────────
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=1024,
                    temperature=0.4,       # controlled, professional outputs
                ),
            )
            reply_text: str = response.text

        except Exception as exc:
            print(f"\n[API ERROR] {exc}\n")
            # Remove the last user turn so history stays consistent
            history.pop()
            continue

        # ── Append model turn to history (strongly typed) ───────────────────
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=reply_text)],
            )
        )

        # ── Display response ─────────────────────────────────────────────────
        print(f"\nStockSage: {reply_text}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
