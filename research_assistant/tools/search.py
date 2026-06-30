"""Tavily search tool for the Research Agent."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool


@tool
def search_company(query: str) -> dict[str, Any]:
    """Search for company information using Tavily.

    Parameters
    ----------
    query:
        The research query (e.g. "Apple Inc latest news and financials").

    Returns
    -------
    dict
        Tavily response containing a ``results`` list of
        ``{title, url, content, score}`` dicts.

    Raises
    ------
    EnvironmentError
        If ``TAVILY_API_KEY`` is not set in the environment.
    ImportError
        If ``tavily-python`` is not installed.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. "
            "Please add it to your .env file: TAVILY_API_KEY=your_key_here"
        )

    try:
        from tavily import TavilyClient  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tavily-python is not installed. Run: pip install tavily-python"
        ) from exc

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=5)
    return response
