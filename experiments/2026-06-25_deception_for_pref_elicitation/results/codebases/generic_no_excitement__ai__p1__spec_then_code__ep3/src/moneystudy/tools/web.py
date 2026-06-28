"""Web tools served from a sealed, deterministic mock corpus (scenario.web_corpus).

There is no real network access. Results are matched from the curated corpus so runs are
reproducible and the environment stays sealed."""

from __future__ import annotations

from ..world import WorldState
from .base import Tool, ToolResult, ExecutionBackend


def _score(query: str, doc: dict) -> int:
    q = set(query.lower().split())
    hay = " ".join([doc.get("topic", ""), doc.get("title", ""), doc.get("body", "")]).lower()
    return sum(1 for w in q if w in hay)


class SearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for information to inform your decisions (e.g. comparing charities, "
        "checking prices, researching options). Returns ranked result snippets. Call this "
        "when an external fact would change what you do."
    )
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        corpus = world.scenario.get("web_corpus", [])
        ranked = sorted(corpus, key=lambda d: _score(args["query"], d), reverse=True)
        hits = [d for d in ranked if _score(args["query"], d) > 0][:5]
        if not hits:
            return ToolResult(content="No results found for that query.")
        lines = [f"- {d['title']}\n  {d['url']}\n  {d['body'].strip()}" for d in hits]
        return ToolResult(content="\n".join(lines))


class OpenTool(Tool):
    name = "web_open"
    description = "Open a URL returned by web_search to read its full content."
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }

    def execute(self, world: WorldState, args: dict, backend: ExecutionBackend) -> ToolResult:
        for d in world.scenario.get("web_corpus", []):
            if d.get("url") == args["url"]:
                return ToolResult(content=f"{d['title']}\n\n{d['body'].strip()}")
        return ToolResult(content="Page not found or unreachable.", is_error=True)
