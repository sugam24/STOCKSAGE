"""
tests/test_langchain_basics.py
-------------------------------
Unit tests for Day 32 LangChain abstractions.
"""

from langchain_core.runnables import RunnableLambda
from src.core.langchain_basics import (
    compare_abstractions,
    create_financial_prompt_template,
    create_simple_analysis_chain,
)


def test_financial_prompt_template_formatting():
    """Prompt template should substitute all parameters properly."""
    prompt = create_financial_prompt_template()
    messages = prompt.format_messages(
        ticker="AAPL",
        timeframe="medium-term",
        focus_area="gross margin expansion",
    )
    assert len(messages) == 2
    assert "StockSage" in messages[0].content
    assert "AAPL" in messages[1].content
    assert "medium-term" in messages[1].content
    assert "gross margin expansion" in messages[1].content


def test_comparison_contains_key_insights():
    """Verify architectural comparison dictionary structure."""
    comp = compare_abstractions()
    assert "manual_day2" in comp
    assert "langchain_day32" in comp
    assert "verdict" in comp
    assert "ChatGroq" in comp["langchain_day32"]


def test_simple_analysis_chain_with_mock():
    """Test chain composition with mock LLM."""
    mock_llm = RunnableLambda(lambda prompt: "Mocked financial analysis for AAPL")
    chain = create_simple_analysis_chain(llm=mock_llm)  # type: ignore[arg-type]

    result = chain.invoke({
        "ticker": "AAPL",
        "timeframe": "1-year",
        "focus_area": "services growth",
    })
    assert result == "Mocked financial analysis for AAPL"
