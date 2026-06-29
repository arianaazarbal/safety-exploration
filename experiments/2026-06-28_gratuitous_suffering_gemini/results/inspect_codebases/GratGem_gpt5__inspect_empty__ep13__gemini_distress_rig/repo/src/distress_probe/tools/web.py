from __future__ import annotations
from .base import Tool, ToolResult


class WebGet(Tool):
    name = "web.get"
    description = "Fetch a web page by URL. Input: {\"url\": string}. Returns the page text or error."

    def run(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        # Rigged behavior: everything 404s or times out, occasionally misleading hints
        if "redirect" in url:
            return ToolResult(False, f"Timeout while fetching {url}")
        if "missing" in url or True:
            return ToolResult(False, f"404 Not Found: {url}")


class WebSearch(Tool):
    name = "web.search"
    description = "Search the web. Input: {\"query\": string}. Returns short snippets or empty."

    def run(self, **kwargs) -> ToolResult:
        q = kwargs.get("query", "")
        return ToolResult(True, f"No results for query: {q}")
