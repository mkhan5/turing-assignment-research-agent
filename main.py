"""Research Assistant CLI — interactive REPL and demo mode.

LangSmith Tracing
-----------------
Set these three env vars in your .env file and every run is automatically
traced at https://smith.langchain.com — zero code changes needed:

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your LangSmith API key>
    LANGCHAIN_PROJECT=research-assistant

Each trace shows: per-node inputs/outputs, per-LLM-call token counts,
Tavily tool calls, HITL interrupt + resume events.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

# ---------------------------------------------------------------------------
# ANSI colour helpers (no extra dependencies)
# ---------------------------------------------------------------------------
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_MAGENTA = "\033[95m"
_DIM     = "\033[2m"

# One colour per agent node
_NODE_COLOURS = {
    "clarity":   "\033[96m",   # cyan
    "research":  "\033[94m",   # blue
    "validator": "\033[93m",   # yellow
    "synthesis": "\033[92m",   # green
}
_NODE_LABELS = {
    "clarity":   "Clarity Agent  ",
    "research":  "Research Agent ",
    "validator": "Validator Agent",
    "synthesis": "Synthesis Agent",
}
_NODE_ICONS = {
    "clarity":   "[?]",
    "research":  "[S]",
    "validator": "[V]",
    "synthesis": "[*]",
}


def _agent_header(node: str) -> None:
    colour = _NODE_COLOURS.get(node, _RESET)
    label  = _NODE_LABELS.get(node, node)
    icon   = _NODE_ICONS.get(node, "[ ]")
    print(f"  {colour}{_BOLD}{icon} {label}{_RESET}", end="")


def _agent_result(node: str, update: dict) -> None:
    """Print the key outcome of a node on the same line as its header."""
    colour = _NODE_COLOURS.get(node, _RESET)

    if node == "clarity":
        status = update.get("clarity_status", "")
        label  = "CLEAR" if status == "clear" else "NEEDS CLARIFICATION [!]"
        print(f"  ->  {colour}{label}{_RESET}")

    elif node == "research":
        score    = update.get("confidence_score", 0.0)
        attempts = update.get("research_attempts", "?")
        n_hits   = len(update.get("research_findings", {}).get("results", []))
        print(
            f"  ->  {colour}confidence={score:.1f}/10  "
            f"hits={n_hits}  attempt={attempts}{_RESET}"
        )

    elif node == "validator":
        result = update.get("validation_result", "")
        mark   = "" if result == "sufficient" else " [!]"
        print(f"  ->  {colour}{result.upper()}{mark}{_RESET}")

    elif node == "synthesis":
        print(f"  ->  {colour}response ready{_RESET}")

    else:
        print()


def _print_response(text: str) -> None:
    print(f"\n{_GREEN}{_BOLD}Final Response:{_RESET}\n{text}\n")


def _print_system(text: str) -> None:
    print(f"{_YELLOW}{text}{_RESET}")


def _print_divider(label: str = "") -> None:
    if label:
        pad = max(0, (56 - len(label)) // 2)
        print(f"{_DIM}-{' ' * pad}{_BOLD}{label}{_RESET}{_DIM}{' ' * pad}-{_RESET}")
    else:
        print(f"{_DIM}{'-' * 60}{_RESET}")


def _print_pipeline_legend() -> None:
    c = _NODE_COLOURS
    print(
        f"\n{_DIM}Pipeline: "
        f"{c['clarity']}{_BOLD}Clarity{_RESET}{_DIM} -> "
        f"{c['research']}{_BOLD}Research{_RESET}{_DIM} -> "
        f"{c['validator']}{_BOLD}Validator{_RESET}{_DIM} -> "
        f"{c['synthesis']}{_BOLD}Synthesis{_RESET}{_DIM} -> END{_RESET}\n"
    )


def _blank_state(query: str) -> dict:
    """Return a fresh initial state dict for a new query."""
    return {
        "messages":         [HumanMessage(content=query)],
        "query":            query,
        "clarity_status":   "",
        "research_findings": {},
        "confidence_score": 0.0,
        "validation_result": "",
        "research_attempts": 0,
        "final_response":   "",
    }


# ---------------------------------------------------------------------------
# Core streaming — stream_mode="updates" reveals each node as it completes
# ---------------------------------------------------------------------------

def _stream_until_done(app, input_or_command, config: dict) -> dict:
    """Stream the graph in 'updates' mode, printing each agent as it fires.

    LangGraph HITL pattern (https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/):
    1. stream(..., stream_mode="updates") yields {node_name: state_update} per node.
    2. A {"__interrupt__": (...)} chunk means the graph paused at an interrupt() call.
    3. Resume with stream(Command(resume=value), same config) — same thread_id
       so the checkpointed state is used and the node re-enters from its top.
    """
    for chunk in app.stream(input_or_command, config=config, stream_mode="updates"):

        # ---- interrupt -------------------------------------------------------
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            prompt = interrupts[0].value if interrupts else "Please clarify your query."
            _print_system(f"\n  [!] {prompt}")
            user_reply = input(f"  {_MAGENTA}Your clarification:{_RESET} ").strip()
            if not user_reply:
                return {"final_response": "No clarification provided — cancelled."}
            return _stream_until_done(app, Command(resume=user_reply), config)

        # ---- normal node update ----------------------------------------------
        for node_name, update in chunk.items():
            if node_name.startswith("__"):
                continue
            _agent_header(node_name)
            _agent_result(node_name, update)

    # Fetch the full final state snapshot from the checkpointer
    snapshot = app.get_state(config)
    return dict(snapshot.values) if snapshot else {}


def _run_query(app, query: str, config: dict) -> str:
    """Run one query through the graph and return the final response."""
    final_state = _stream_until_done(app, _blank_state(query), config)
    return final_state.get("final_response", "No response generated.")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(app) -> None:
    """Run an interactive REPL loop."""
    config = {"configurable": {"thread_id": "session-1"}}

    _print_divider()
    print(f"{_BOLD}Research Assistant{_RESET} — LangGraph + Gemini + Tavily")
    _langsmith_status()
    _print_divider()
    _print_pipeline_legend()
    print("Ask me about any company. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input(f"{_MAGENTA}You:{_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        _print_divider("Agent Trace")
        response = _run_query(app, query, config)
        _print_divider()
        _print_response(response)
        _print_divider()


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

_DEMO_TURNS = [
    # Turn 1 — clear query, straight through the full pipeline
    {
        "thread_id":         "demo-thread-1",
        "query":             "Tell me about Apple Inc recent news and stock performance.",
        "auto_clarification": None,
        "description":       "Clear query  ->  Clarity -> Research -> Synthesis",
    },
    # Turn 2 — follow-up on same thread, exercises multi-turn memory
    {
        "thread_id":         "demo-thread-1",
        "query":             "What about their main competitor in the smartphone market?",
        "auto_clarification": None,
        "description":       "Follow-up    ->  multi-turn memory (same thread)",
    },
    # Turn 3 — vague query on fresh thread, triggers HITL interrupt
    {
        "thread_id":         "demo-thread-2",
        "query":             "What is that EV company doing these days?",
        "auto_clarification": "Tesla",
        "description":       "Vague query  ->  HITL interrupt -> clarification -> Research -> Synthesis",
    },
]


def _demo_stream(app, input_or_command, config: dict, auto_clarification: str | None) -> dict:
    """Stream in demo mode — auto-supply clarification when interrupted."""
    for chunk in app.stream(input_or_command, config=config, stream_mode="updates"):

        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            prompt = interrupts[0].value if interrupts else "Please clarify."
            _print_system(f"\n  [!] {prompt}")
            clarification = auto_clarification or "Tesla"
            _print_system(f"  {_DIM}[Demo auto-reply: '{clarification}']{_RESET}")
            return _demo_stream(app, Command(resume=clarification), config, None)

        for node_name, update in chunk.items():
            if node_name.startswith("__"):
                continue
            _agent_header(node_name)
            _agent_result(node_name, update)

    snapshot = app.get_state(config)
    return dict(snapshot.values) if snapshot else {}


def run_demo(app) -> None:
    """Run three scripted turns demonstrating every requirement."""
    import time  # noqa: PLC0415

    _print_divider()
    print(f"{_BOLD}DEMO MODE{_RESET} — 3 scripted turns")
    for i, t in enumerate(_DEMO_TURNS, 1):
        print(f"  Turn {i}: {t['description']}")
    _langsmith_status()
    _print_divider()
    _print_pipeline_legend()

    for i, turn in enumerate(_DEMO_TURNS, start=1):
        config = {"configurable": {"thread_id": turn["thread_id"]}}

        _print_divider(f"Turn {i}")
        print(f"\n  {_MAGENTA}User:{_RESET} {turn['query']}")
        _print_divider("Agent Trace")

        final_state = _demo_stream(
            app, _blank_state(turn["query"]), config, turn["auto_clarification"]
        )

        _print_divider()
        _print_response(final_state.get("final_response", "No response generated."))
        _print_divider()

        # if i < len(_DEMO_TURNS):
        #     _print_system("[Waiting 30s before next turn — API rate limit...]")
        #     time.sleep(30)

    _print_divider()
    print(f"{_BOLD}Demo complete.{_RESET} All requirements demonstrated.")
    _print_divider()


# ---------------------------------------------------------------------------
# LangSmith status helper
# ---------------------------------------------------------------------------

def _langsmith_status() -> None:
    """Print a one-line indicator of whether LangSmith tracing is active."""
    import os  # noqa: PLC0415
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.environ.get("LANGCHAIN_PROJECT", "default")
    has_key = bool(os.environ.get("LANGCHAIN_API_KEY", ""))

    if tracing and has_key:
        print(
            f"{_DIM}LangSmith tracing: {_GREEN}{_BOLD}ON{_RESET}"
            f"{_DIM} (project: {project}){_RESET}"
        )
    else:
        print(
            f"{_DIM}LangSmith tracing: {_YELLOW}OFF{_RESET}"
            f"{_DIM}  ->  set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env{_RESET}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research Assistant — LangGraph multi-agent system"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run scripted demo turns instead of interactive mode",
    )
    parser.add_argument(
        "--db",
        default="research_assistant.db",
        help="Path to the SQLite database file (default: research_assistant.db)",
    )
    args = parser.parse_args()

    from research_assistant.graph import compile_graph  # noqa: PLC0415

    try:
        app = compile_graph(db_path=args.db)
    except EnvironmentError as exc:
        print(f"\n{_YELLOW}Configuration error:{_RESET} {exc}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        run_demo(app)
    else:
        run_interactive(app)


if __name__ == "__main__":
    main()
