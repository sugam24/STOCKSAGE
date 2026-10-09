"""
agents/critic_node.py
----------------------
Day 49: Self-Reflection Critic Node (Reflexion Pattern).

Evaluates the synthesized research report produced by the Analyst Agent against
an editorial rubric (completeness, source citations, balanced bull/bear analysis,
and quantitative risk accuracy). If the score is below 4, returns actionable critique
and triggers an iterative revision loop (capped at max 2 iterations).

Usage:
    uv run python -m src.agents.critic_node
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from src.graph.state import StockSageState

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_REVISION_CYCLES = 2


class CriticReview(BaseModel):
    """
    Step 222: Structured evaluation rubric output from the Critic node.
    """

    score: int = Field(..., ge=1, le=5, description="Overall institutional quality score (1 to 5).")
    is_passing: bool = Field(..., description="True if score >= 4, indicating publication-ready quality.")
    completeness_score: int = Field(..., ge=1, le=5, description="Rating for coverage of fundamentals, news, and risk.")
    citation_score: int = Field(..., ge=1, le=5, description="Rating for citing specific facts, numbers, and sources.")
    actionable_feedback: str = Field(..., description="Concrete, constructive suggestions for the Analyst Agent.")


CRITIC_SYSTEM_PROMPT = """\
You are the Managing Editor and Senior Risk Officer at StockSage.
Your role is to rigorously audit equity research drafts against our editorial standard.

Grading Rubric:
- 5 (Institutional Grade): Clear executive synthesis with specific numbers, balanced bull and bear cases, explicit citations/data points, and accurate risk metrics.
- 4 (Publication Ready): Well-structured, specific, minor room for polish.
- 3 (Borderline): Generic observations, lacks specific figures (RSI, P/E, VaR), or one-sided argument.
- 1–2 (Failed Draft): Incomplete, missing bull or bear section, vague generalities, or fails to address company realities.

You MUST respond strictly with a valid JSON object matching:
{
  "score": 1 to 5,
  "is_passing": true/false (true if score >= 4),
  "completeness_score": 1 to 5,
  "citation_score": 1 to 5,
  "actionable_feedback": "Specific instructions on what to fix or expand."
}
"""


def _review_report_sync(
    report_text: str,
    ticker: str,
    model: str = DEFAULT_MODEL,
) -> CriticReview:
    """Synchronous worker that calls Groq with JSON mode to grade the report."""
    if not report_text.strip():
        return CriticReview(
            score=1,
            is_passing=False,
            completeness_score=1,
            citation_score=1,
            actionable_feedback="The draft report was empty. Produce a full structured report.",
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)
    prompt = (
        f"Audit this research draft for ticker: {ticker}\n\n"
        f"--- DRAFT START ---\n{report_text}\n--- DRAFT END ---\n\n"
        "Score the report 1–5 and provide actionable editorial critique."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content or "{}"
    data = json.loads(raw_json)

    score = int(data.get("score", 3))
    data["score"] = max(1, min(5, score))
    data["is_passing"] = data["score"] >= 4
    data["completeness_score"] = int(data.get("completeness_score", data["score"]))
    data["citation_score"] = int(data.get("citation_score", data["score"]))
    if not data.get("actionable_feedback"):
        data["actionable_feedback"] = "Add more quantitative data points and specific filing facts."

    return CriticReview.model_validate(data)


async def critic_node(state: StockSageState | dict[str, Any]) -> dict[str, Any]:
    """
    Step 222–224: Async LangGraph node for the Critic.

    Evaluates state['final_report'], attaches critique, and increments revision_count.
    """
    report = state.get("final_report", "")
    ticker = state.get("ticker", "UNKNOWN")
    current_revisions = state.get("revision_count", 0)

    try:
        review = await asyncio.to_thread(_review_report_sync, report, ticker)
        new_revision_count = current_revisions + (1 if not review.is_passing else 0)

        critique_summary = (
            f"[Critic Score: {review.score}/5 (Passing: {review.is_passing})] "
            f"{review.actionable_feedback}"
        )

        return {
            "critique": critique_summary,
            "revision_count": new_revision_count,
        }

    except Exception as exc:
        return {
            "critique": f"Critic skipped due to error: {exc}",
            "revision_count": current_revisions + 1,
            "error_log": [f"Critic error: {exc}"],
        }


def route_from_critic(state: StockSageState) -> str:
    """
    Step 223 & 224: Route back to analyst_agent if weak and below max iterations,
    otherwise proceed to finalize_turn.
    """
    critique = state.get("critique", "")
    revisions = state.get("revision_count", 0)

    # If failing score and hasn't exceeded 2 revision cycles, loop back to analyst
    is_failing = "Passing: False" in critique or "[Critic Score: 1" in critique or "[Critic Score: 2" in critique or "[Critic Score: 3" in critique

    if is_failing and revisions <= MAX_REVISION_CYCLES:
        return "analyst_agent"

    return "finalize_turn"


if __name__ == "__main__":
    print("=" * 65)
    print("  StockSage — Day 49: Self-Reflection Critic Node Test")
    print("=" * 65)

    weak_report = (
        "# Report for AAPL\n"
        "Apple makes iPhones. The stock might go up or down. "
        "It seems like a good company."
    )

    async def _test():
        print("Evaluating a deliberately weak draft report...")
        state: StockSageState = {
            "ticker": "AAPL",
            "final_report": weak_report,
            "critique": "",
            "revision_count": 0,
            "error_log": [],
        }
        res = await critic_node(state)
        print("\nCritic Result:")
        print(f"  • Critique: {res.get('critique')}")
        print(f"  • Revision Count: {res.get('revision_count')}")
        next_step = route_from_critic({**state, **res})
        print(f"  • Next Routing Step: {next_step} (loop back to analyst_agent!)")

    asyncio.run(_test())
