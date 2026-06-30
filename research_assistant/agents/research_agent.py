"""Research Agent — performs live web search and scores result confidence."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from research_assistant.state import ResearchState
from research_assistant.tools.search import search_company


class ResearchOutput(BaseModel):
    """Structured output from the Research Agent."""

    confidence_score: float = Field(
        ge=0.0,
        le=10.0,
        description="How well the search results answer the user's query (0-10).",
    )
    summary: str = Field(
        description="Brief summary of what was found in the search results."
    )


_SYSTEM_PROMPT = """\
You are a research analyst assistant. Your job is to research a company query.

Use the search_company tool to look up information, then:
1. Review the search results and extract information relevant to the query.
2. Assign a confidence_score between 0 and 10 that reflects how completely and
   accurately the search results answer the user's question:
   - 0-3: Very little or no relevant information found
   - 4-5: Some relevant information but major gaps remain
   - 6-7: Good coverage with minor gaps
   - 8-10: Comprehensive, high-quality information found
3. Write a brief summary (2-4 sentences) of the key findings.

Be honest about gaps — do not inflate the confidence score.
"""


class ResearchAgent:
    """Agent that runs a Tavily search via tool calling and scores the results."""

    def __init__(self, llm) -> None:
        self._llm_with_tools = llm.bind_tools([search_company])
        self._scoring_chain = llm.with_structured_output(ResearchOutput)

    def run(self, state: ResearchState) -> dict[str, Any]:
        """Execute a live search via tool calling and evaluate result quality.

        Parameters
        ----------
        state:
            Current graph state.

        Returns
        -------
        dict
            Partial state update with ``research_findings``,
            ``confidence_score``, and incremented ``research_attempts``.
        """
        query = state["query"]

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Research the following query: {query}"),
        ]

        # Let the LLM decide whether and how to call search_company
        response = self._llm_with_tools.invoke(messages)

        # Dispatch any tool calls the model requested
        raw_findings: dict[str, Any] = {}
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "search_company":
                    raw_findings = search_company.invoke(tool_call["args"])
                    # Append the assistant message + tool result so the scorer
                    # has full context
                    messages.append(response)
                    messages.append(
                        ToolMessage(
                            content=_format_results(raw_findings),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    break  # one search per turn is sufficient

        # Score the findings with structured output
        scoring_messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User query: {query}\n\n"
                    f"Search results:\n{_format_results(raw_findings)}"
                )
            ),
        ]
        result: ResearchOutput = self._scoring_chain.invoke(scoring_messages)

        return {
            "research_findings": raw_findings,
            "confidence_score": result.confidence_score,
            "research_attempts": state.get("research_attempts", 0) + 1,
        }


def _format_results(tavily_response: dict[str, Any]) -> str:
    """Format Tavily results into a readable string for the LLM."""
    results = tavily_response.get("results", [])
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "No content")
        lines.append(f"[{i}] {title}\n    URL: {url}\n    {content}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_research(state: ResearchState) -> Literal["validator", "synthesis"]:
    """Conditional edge: route based on confidence_score."""
    if state["confidence_score"] < 6.0:
        return "validator"
    return "synthesis"
