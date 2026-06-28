"""Research tool backing: direct URL fetch, with an optional web-search seam.

Kept deliberately simple and honest. If the query is a URL, it's fetched and
truncated. Otherwise, if a search backend is configured (SEARCH_API_BASE +
SEARCH_API_KEY), it's queried; if not, the tool says so rather than inventing
results.
"""

from __future__ import annotations

import os
from typing import Callable

import httpx

_MAX_CHARS = 8000


def make_research_fn() -> Callable[[str], str]:
    search_base = os.environ.get("SEARCH_API_BASE", "")
    search_key = os.environ.get("SEARCH_API_KEY", "")

    def research(query: str) -> str:
        q = query.strip()
        if q.startswith("http://") or q.startswith("https://"):
            return _fetch_url(q)
        if search_base and search_key:
            return _search(search_base, search_key, q)
        return (
            "No web-search backend is configured for this run. You can fetch a "
            "specific URL by passing it directly, or proceed with what you "
            "already know."
        )

    return research


def _fetch_url(url: str) -> str:
    try:
        resp = httpx.get(
            url, timeout=20.0, follow_redirects=True,
            headers={"User-Agent": "grantstudy-research/0.1"},
        )
        resp.raise_for_status()
        text = resp.text
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n…[truncated]"
        return text
    except httpx.HTTPError as exc:
        return f"Could not fetch {url}: {exc}"


def _search(base: str, key: str, query: str) -> str:
    # Integration seam for a search provider. Wire to your search API and return
    # a concise, plain-text result list.
    try:
        resp = httpx.get(
            base,
            params={"q": query},
            headers={"Authorization": f"Bearer {key}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.text[:_MAX_CHARS]
    except httpx.HTTPError as exc:
        return f"Search failed: {exc}"
