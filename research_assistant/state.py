"""Shared state schema for the Research Assistant graph."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """State that flows through every node in the research assistant graph.

    Fields
    ------
    messages:
        Full conversation history as LangChain message objects.
        Uses the ``add_messages`` reducer so each node can append
        without overwriting prior history.
    query:
        The current user query being processed.
    clarity_status:
        Set by the Clarity Agent: ``"clear"`` or ``"needs_clarification"``.
    research_findings:
        Raw results returned by the Research Agent (Tavily payload).
    confidence_score:
        0–10 score assigned by the Research Agent via the LLM.
    validation_result:
        Set by the Validator Agent: ``"sufficient"`` or ``"insufficient"``.
    research_attempts:
        Incremented by the Research Agent each time it runs.
        Caps the Validator → Research feedback loop at 3 attempts.
    final_response:
        The polished, user-facing answer produced by the Synthesis Agent.
    """

    messages: Annotated[list, add_messages]
    query: str
    clarity_status: str
    research_findings: dict
    confidence_score: float
    validation_result: str
    research_attempts: int
    final_response: str
