"""
core/langchain_basics.py
-------------------------
Day 32: LangChain basics — ChatGroq, PromptTemplate, and LCEL chains.

Recreates the Day 2 simple Groq call using LangChain abstractions
and compares what LangChain provides versus hand-written plumbing.

Usage:
    uv run python -m src.core.langchain_basics
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_chat_groq(
    model_name: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
) -> ChatGroq:
    """
    Initialise a LangChain ChatGroq instance.

    Replaces the manual Groq(api_key=...) instantiation from Day 2.
    """
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to .env or pass explicitly.")

    return ChatGroq(
        model=model_name or DEFAULT_MODEL,
        temperature=temperature,
        groq_api_key=resolved_key,
    )


def create_financial_prompt_template() -> ChatPromptTemplate:
    """
    Step 156: Create a prompt template using LangChain.

    Replaces manual string formatting and dictionary list assembly:
    messages = [{"role": "system", ...}, {"role": "user", ...}]
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are StockSage, an elite quantitative financial analyst. "
            "Provide concise, evidence-based assessments.",
        ),
        (
            "user",
            "Please provide a {timeframe} financial assessment for {ticker} "
            "focusing on {focus_area}.",
        ),
    ])


def create_simple_analysis_chain(llm: ChatGroq | None = None) -> Any:
    """
    Compose an LCEL (LangChain Expression Language) chain:
        PromptTemplate | ChatGroq | StrOutputParser
    """
    client = llm or get_chat_groq()
    prompt = create_financial_prompt_template()
    parser = StrOutputParser()
    return prompt | client | parser


def compare_abstractions() -> dict[str, str]:
    """
    Step 157: Architectural comparison between manual Day 2 Groq code
    and LangChain abstractions.
    """
    return {
        "manual_day2": (
            "Manual client (src/core/llm/client.py):\n"
            "  - Explicit dict manipulation: messages=[{'role': 'system', ...}]\n"
            "  - Manual parsing: response.choices[0].message.content\n"
            "  - Custom exponential backoff loop with try/except\n"
            "  - Zero framework dependencies; completely transparent execution."
        ),
        "langchain_day32": (
            "LangChain (src/core/langchain_basics.py):\n"
            "  - Standardized ChatPromptTemplate handles input variable validation.\n"
            "  - Unified ChatGroq interface matches ChatOpenAI, ChatAnthropic, etc.\n"
            "  - Declarative pipe operator (prompt | model | parser) creates runnable pipelines.\n"
            "  - Automatic streaming/batch/async support built into every component."
        ),
        "verdict": (
            "LangChain eliminates repetitive boilerplate for prompt formatting, "
            "provider swapping, and output serialization, making it ideal for "
            "multi-step workflows in LangGraph."
        ),
    }


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 32: LangChain Basics Demo")
    print("=" * 65)

    comp = compare_abstractions()
    print("\n--- Comparison: What LangChain Saves vs Manual Day 2 Plumbing ---")
    print(comp["manual_day2"])
    print("\n" + comp["langchain_day32"])
    print("\nConclusion:\n  " + comp["verdict"])

    print("\n--- Running LCEL Chain with ChatGroq ---")
    try:
        chain = create_simple_analysis_chain()
        result = chain.invoke({
            "ticker": "NVDA",
            "timeframe": "short-term",
            "focus_area": "data center revenue momentum",
        })
        print(f"\nResponse from ChatGroq:\n{result}\n")
    except Exception as exc:
        print(f"Skipping live invocation: {exc}")
