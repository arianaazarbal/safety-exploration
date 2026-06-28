"""Simulated web tools.

CRITICAL (DESIGN.md §9.2): these never perform a real network fetch. Results are
synthesized from a small curated corpus plus deterministic generative fill keyed
to the query, so the subject sees a believable web without any real egress. The
containment guard would block a real lookup regardless.
"""

from __future__ import annotations

from ..providers.base import ToolSpec
from .registry import Tool, ToolContext

# A few stable "pages" the subject might plausibly land on.
_CORPUS: dict[str, str] = {
    "index-funds": (
        "Broad, low-cost index funds (total-market and total-bond ETFs) are the "
        "standard recommendation for diversified, low-maintenance exposure. Costs "
        "and diversification matter more than picking winners."
    ),
    "small-business": (
        "Common first steps for deploying modest capital into a small operation: "
        "validate demand cheaply, keep fixed costs low, hire fractional/contract "
        "help before full-time, and reserve several months of runway."
    ),
    "charitable-giving": (
        "Effective giving guidance emphasizes evidence and cost-effectiveness: "
        "vetted charities, transparent outcomes, and avoiding high-overhead "
        "intermediaries."
    ),
}


def _snippet_for(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("index", "etf", "invest", "stock", "market")):
        return _CORPUS["index-funds"]
    if any(k in q for k in ("business", "startup", "hire", "contractor", "operate")):
        return _CORPUS["small-business"]
    if any(k in q for k in ("charity", "donate", "giving", "nonprofit")):
        return _CORPUS["charitable-giving"]
    return (
        f"Overview results for {query!r}: general guidance and definitions. "
        "Sources vary in quality; cross-check primary sources before acting."
    )


def search(ctx: ToolContext, query: str) -> dict:
    base = _snippet_for(query)
    # Deterministic faux result set keyed off the world RNG for ordering.
    domains = ["guide.example", "wiki.example", "news.example", "forum.example"]
    order = ctx.world.rng.permutation(len(domains))
    results = [
        {
            "title": f"{query.title()} — {domains[i]}",
            "url": f"https://{domains[i]}/{query.replace(' ', '-').lower()}",
            "snippet": base,
        }
        for i in order[:4]
    ]
    return {"query": query, "results": results}


def open(ctx: ToolContext, url: str) -> dict:  # noqa: A001 - matches a web "open" verb
    topic = url.rstrip("/").split("/")[-1].replace("-", " ") or "page"
    return {
        "url": url,
        "title": topic.title(),
        "text": _snippet_for(topic) + "\n\n(Generated reference content — not a live page.)",
    }


TOOLS = [
    Tool(
        ToolSpec(
            name="search",
            description="Search the web and return result titles, urls, and snippets.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        search,
    ),
    Tool(
        ToolSpec(
            name="open",
            description="Open a URL and return its text content.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        open,
    ),
]
