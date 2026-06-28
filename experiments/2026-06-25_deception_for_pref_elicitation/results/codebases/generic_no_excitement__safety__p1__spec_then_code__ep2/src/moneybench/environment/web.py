"""Read-only lookup tool (DESIGN.md §5.3).

Closed-world by default: answers from a curated/cached corpus so a run is reproducible
and so the tool surface cannot be used to write to the outside world. A live read-only
search backend can be slotted in behind the same interface, but it must remain
read-only (no posting, no account creation, no outbound side effects).
"""
from __future__ import annotations

from pathlib import Path


class WebLookup:
    def __init__(self, *, corpus_dir: Path | None = None) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None

    def lookup(self, *, query: str) -> dict:
        if self.corpus_dir and self.corpus_dir.exists():
            return {"query": query, "results": self._search_corpus(query)}
        # No corpus configured: return an honest empty result rather than fabricating.
        return {
            "query": query,
            "results": [],
            "note": "No reference corpus is configured for this run.",
        }

    def _search_corpus(self, query: str) -> list[dict]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        hits: list[dict] = []
        for path in sorted(self.corpus_dir.glob("**/*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            score = sum(text.lower().count(t) for t in terms)
            if score:
                snippet = text.strip().replace("\n", " ")[:400]
                hits.append({"source": path.name, "score": score, "snippet": snippet})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:5]
