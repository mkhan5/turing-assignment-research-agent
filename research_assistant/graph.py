"""Graph assembly — builds and compiles the Research Assistant StateGraph.

LangSmith Tracing
-----------------
LangGraph automatically sends traces to LangSmith when the following env vars
are set in your .env file (zero code changes required):

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your LangSmith API key>
    LANGCHAIN_PROJECT=research-assistant

Every node (clarity, research, validator, synthesis), every LLM call inside
each node, token counts, and Tavily tool calls will appear as a nested trace
tree at https://smith.langchain.com.
"""

from __future__ import annotations

import os
import sqlite3

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from research_assistant.agents.clarity_agent import ClarityAgent, route_after_clarity
from research_assistant.agents.research_agent import ResearchAgent, route_after_research
from research_assistant.agents.synthesis_agent import SynthesisAgent
from research_assistant.agents.validator_agent import ValidatorAgent, route_after_validation
from research_assistant.state import ResearchState


# ---------------------------------------------------------------------------
# LLM singleton — shared across all agents
# ---------------------------------------------------------------------------

def _build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Please add it to your .env file: GOOGLE_API_KEY=your_key_here"
        )
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def compile_graph(db_path: str = "research_assistant.db"):
    """Build and compile the Research Assistant StateGraph.

    The human-in-the-loop interrupt lives inside ``ClarityAgent.run``.
    When the query is ambiguous, ``interrupt()`` pauses the graph at that
    node. The caller detects the ``__interrupt__`` chunk from ``stream()``,
    collects user input, then resumes with ``Command(resume=<value>)``.
    LangGraph re-enters the clarity node; ``interrupt()`` returns the value
    instead of raising, and execution continues normally.

    A checkpointer is required for ``interrupt()`` to work — state must be
    persisted so the graph can be resumed from exactly where it paused.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file used for conversation persistence.
        Use ``:memory:`` for an in-process ephemeral store (tests only).
        Defaults to ``research_assistant.db`` in the current working directory.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph application ready to invoke.
    """
    # Open a persistent SQLite connection (not via context manager) so the
    # checkpointer stays alive for the full duration of the process.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    llm = _build_llm()

    # Instantiate agents
    clarity_agent  = ClarityAgent(llm)
    research_agent = ResearchAgent(llm)
    validator_agent = ValidatorAgent(llm)
    synthesis_agent = SynthesisAgent(llm)

    # Build graph
    builder = StateGraph(ResearchState)

    # Register nodes — no separate interrupt node needed; the interrupt()
    # call lives directly inside ClarityAgent.run.
    builder.add_node("clarity",   clarity_agent.run)
    builder.add_node("research",  research_agent.run)
    builder.add_node("validator", validator_agent.run)
    builder.add_node("synthesis", synthesis_agent.run)

    # Entry point
    builder.add_edge(START, "clarity")

    # Clarity always routes to Research (the interrupt inside the node handles
    # the clarification loop before routing ever runs).
    builder.add_edge("clarity", "research")

    # Research -> Validator or Synthesis based on confidence_score
    builder.add_conditional_edges(
        "research",
        route_after_research,
        {"validator": "validator", "synthesis": "synthesis"},
    )

    # Validator -> Research (loop, max 3 attempts) or Synthesis
    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {"research": "research", "synthesis": "synthesis"},
    )

    # Synthesis is always terminal
    builder.add_edge("synthesis", END)

    # Compile with SQLite checkpointer — required for interrupt() to work
    return builder.compile(checkpointer=checkpointer)
