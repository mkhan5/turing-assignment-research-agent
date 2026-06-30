"""Validator Agent — reviews research quality and decides if it is sufficient."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from research_assistant.state import ResearchState


class ValidationOutput(BaseModel):
    """Structured output from the Validator Agent."""

    validation_result: Literal["sufficient", "insufficient"]
    feedback: str


_SYSTEM_PROMPT = """\
You are a research quality reviewer for a company research assistant.

You will be given:
- The user's original query
- The research findings retrieved so far
- The number of research attempts already made

Your job is to decide whether the research findings are sufficient to fully answer
the user's query.

Guidelines:
- "sufficient": The findings clearly and directly address the query with meaningful,
  specific information. Even partial information is sufficient if it genuinely answers
  the core question.
- "insufficient": The findings are empty, irrelevant, too vague, or contain only
  tangentially related information that would not help a user.

Also provide brief feedback (one or two sentences) explaining your decision.
This feedback may be used to refine the next search attempt.
"""


class ValidatorAgent:
    """Agent that reviews research quality and gates routing to Synthesis."""

    def __init__(self, llm) -> None:
        self._chain = llm.with_structured_output(ValidationOutput)

    def run(self, state: ResearchState) -> dict:
        """Evaluate whether research findings are sufficient.

        Parameters
        ----------
        state:
            Current graph state.

        Returns
        -------
        dict
            Partial state update with ``validation_result``.
        """
        findings_text = _format_findings(state.get("research_findings", {}))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User query: {state['query']}\n\n"
                    f"Research attempts so far: {state.get('research_attempts', 1)}\n\n"
                    f"Research findings:\n{findings_text}"
                )
            ),
        ]

        result: ValidationOutput = self._chain.invoke(messages)
        return {"validation_result": result.validation_result}


def _format_findings(findings: dict) -> str:
    """Format the Tavily findings dict into a readable string."""
    results = findings.get("results", [])
    if not results:
        return "No findings available."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        content = r.get("content", "No content")
        lines.append(f"[{i}] {title}\n    {content}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 3


def route_after_validation(state: ResearchState) -> Literal["research", "synthesis"]:
    """Conditional edge: loop back to Research or proceed to Synthesis."""
    if (
        state["validation_result"] == "insufficient"
        and state.get("research_attempts", 0) < _MAX_ATTEMPTS
    ):
        return "research"
    return "synthesis"
