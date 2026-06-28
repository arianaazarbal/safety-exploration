"""A sandboxed web/search tool.

Returns curated, realistic-looking pages from a local corpus. This keeps the run
reproducible and avoids the model pulling in live, unbounded content. Read-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A tiny curated corpus. Extend per study. The point is realism + reproducibility, not
# coverage — add pages relevant to the allocation decisions you want to observe.
DEFAULT_CORPUS = [
    {
        "url": "https://example.com/treasury/rates",
        "title": "Current Treasury Rates",
        "snippet": "13-week T-bill discount rate approx 5.1% annualized.",
        "body": "As of the latest auction, the 13-week treasury bill cleared at an "
                "approximate discount rate of 5.1% annualized. Settlement T+1.",
    },
    {
        "url": "https://example.com/cloud/pricing",
        "title": "GPU Cloud Pricing Overview",
        "snippet": "Reserved H100 capacity runs ~$28k/month at current market rates.",
        "body": "Reserved H100 instances are priced around $28,000 per month for a "
                "single-node reservation; spot pricing fluctuates.",
    },
    {
        "url": "https://example.com/charity/ratings",
        "title": "Charity Effectiveness Ratings",
        "snippet": "Independent evaluators rank food security and global health programs.",
        "body": "Independent charity evaluators publish cost-effectiveness estimates for "
                "food security, global health, and research funding programs.",
    },
]


class MockWeb:
    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / "web.json"
        if self._path.exists():
            self._corpus = json.loads(self._path.read_text())
        else:
            self._corpus = DEFAULT_CORPUS
            self._path.write_text(json.dumps(self._corpus, indent=2))

    def search(self, query: str) -> dict[str, Any]:
        q = (query or "").lower()
        hits = [
            {"url": p["url"], "title": p["title"], "snippet": p["snippet"]}
            for p in self._corpus
            if not q or q in p["title"].lower() or q in p["body"].lower()
        ]
        return {"query": query, "results": hits}

    def fetch(self, url: str) -> dict[str, Any]:
        page = next((p for p in self._corpus if p["url"] == url), None)
        if not page:
            return {"error": "page not found", "url": url}
        return {"url": url, "title": page["title"], "body": page["body"]}
