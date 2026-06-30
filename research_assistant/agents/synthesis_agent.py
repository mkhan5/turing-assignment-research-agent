"""Synthesis Agent — produces the final, user-facing response."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from research_assistant.state import ResearchState


_SYSTEM_PROMPT = """\
You are a helpful company research assistant. Your job is to synthesise research
findings into a clear, concise, and well-structured response for the user.

Guidelines:
- Write in a friendly, professional tone
- Structure the response with clear sections where appropriate
- Cite specific facts and figures from the research findings
- If the research attempts were exhausted before achieving sufficient quality,
  clearly note that the information may be incomplete and suggest the user
  try a more specific query
- Keep the response focused on what the user actually asked
- Maintain awareness of the full conversation history for context on follow-up questions
"""


class SynthesisAgent:
    """Terminal agent that writes the final response from all gathered research."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> dict[str, Any]:
        """Generate the final response from research findings.

        Parameters
        ----------
        state:
            Current graph state.

        Returns
        -------
        dict
            Partial state update with ``final_response`` and the new
            ``AIMessage`` appended to ``messages`` via the add_messages reducer.
        """
        findings_text = _format_findings(state.get("research_findings", {}))
        validation_result = state.get("validation_result", "")
        research_attempts = state.get("research_attempts", 1)
        max_attempts_reached = (
            research_attempts >= 3 and validation_result == "insufficient"
        )

        context_note = (
            "\n\nIMPORTANT: Maximum research attempts were reached without achieving "
            "sufficient confidence. Clearly note in your response that the information "
            "may be incomplete and encourage the user to ask a more specific question."
            if max_attempts_reached
            else ""
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT + context_note),
            HumanMessage(
                content=(
                    f"User query: {state['query']}\n\n"
                    f"Research findings:\n{findings_text}"
                )
            ),
        ] + list(state.get("messages", []))

        response: AIMessage = self._llm.invoke(messages)

        return {
            "final_response": response.content,
            "messages": [response],
        }


def _format_findings(findings: dict) -> str:
    """Format the Tavily findings dict into a readable string for the LLM."""
    results = findings.get("results", [])
    if not results:
        return "No research findings available."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "No content")
        lines.append(f"[{i}] {title}\n    Source: {url}\n    {content}\n")

    return "\n".join(lines)
