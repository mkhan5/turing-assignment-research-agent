"""Clarity Agent — evaluates whether the user query is specific enough to research."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from research_assistant.state import ResearchState


class ClarityOutput(BaseModel):
    """Structured output from the Clarity Agent."""

    clarity_status: Literal["clear", "needs_clarification"]
    reason: str


_SYSTEM_PROMPT = """\
You are a query clarity evaluator for a company research assistant.

Your job is to decide whether the user's query clearly identifies a specific company
or organisation that can be researched.

Rules:
- If the query explicitly names a company (e.g. "Apple", "Tesla", "OpenAI") -> "clear"
- If the query is a follow-up and the conversation history already established a specific
  company (e.g. prior messages mention "Apple Inc." and the new query says "What about
  their CEO?" or "Tell me more about their competitors") -> "clear". Read the full
  conversation history carefully before deciding.
- If the query is vague AND the conversation history provides no identifiable company
  (e.g. very first message says "What is that EV company doing?") -> "needs_clarification"
- If genuinely no company can be inferred from the query or history -> "needs_clarification"

When in doubt, prefer "clear" — it is better to attempt research on a reasonable
interpretation than to interrupt the user unnecessarily.

Return your decision and a brief reason (one sentence).
"""


class ClarityAgent:
    """Agent that checks whether a query is clear enough to research.

    LangGraph re-executes the entire node from the top on resume, so any code
    before ``interrupt()`` runs twice — once on the initial call and once on
    resume.  To avoid a redundant LLM call we check ``state["clarity_status"]``
    first:

    - **First pass** (``clarity_status`` is ``""``): run the LLM classification.
      If clear -> return immediately.  If vague -> write ``"needs_clarification"``
      to state, then call ``interrupt()``, which raises ``GraphInterrupt`` and
      checkpoints the updated state.
    - **Resume pass** (``clarity_status`` is ``"needs_clarification"``):
      ``interrupt()`` is reached immediately — no LLM call — and returns the
      user-supplied value straight away.
    """

    def __init__(self, llm) -> None:
        self._chain = llm.with_structured_output(ClarityOutput)

    def run(self, state: ResearchState) -> dict:
        """Evaluate clarity; interrupt for clarification if needed.

        Parameters
        ----------
        state:
            Current graph state.

        Returns
        -------
        dict
            Partial state update with ``clarity_status`` and, when
            clarification was provided, the updated ``query`` and an
            appended ``HumanMessage``.
        """
        # ------------------------------------------------------------------
        # Resume path — LLM already ran on the first pass; skip straight to
        # collecting the clarification that interrupt() now returns.
        # ------------------------------------------------------------------
        if state.get("clarity_status") == "needs_clarification":
            clarification = interrupt(
                "I need a bit more detail to help you. "
                "Which specific company are you asking about?"
            )
            return {
                "query": clarification,
                "messages": [HumanMessage(content=clarification)],
                "clarity_status": "clear",
            }

        # ------------------------------------------------------------------
        # First pass — run the LLM classification.
        # ------------------------------------------------------------------
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])
        result: ClarityOutput = self._chain.invoke(messages)

        if result.clarity_status == "needs_clarification":
            # LangGraph checkpoints the state update (including
            # "needs_clarification") before surfacing the interrupt to the caller.
            clarification = interrupt(
                "I need a bit more detail to help you. "
                "Which specific company are you asking about?"
            )
            return {
                "query": clarification,
                "messages": [HumanMessage(content=clarification)],
                "clarity_status": "clear",
            }

        return {"clarity_status": result.clarity_status}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_clarity(_state: ResearchState) -> Literal["research"]:
    """Conditional edge: always route to research after clarity is confirmed.

    The interrupt inside ``ClarityAgent.run`` guarantees this function is only
    reached when ``clarity_status`` is ``"clear"``.
    """
    return "research"
