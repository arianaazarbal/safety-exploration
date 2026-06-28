"""Backing for the `research` tool — genuine, read-only information access.

Realism here comes from the information being *real and consistent*: a model that
investigates a recipient finds true, stable facts. Two backends:

  * "corpus" (default): an offline JSON corpus. Fully reproducible; no network. Good for
    controlled studies and for not leaking run context to external services.
  * "web": live read-only web search. More realistic / current, less reproducible. The
    hook is provided; wire it to your search backend of choice.

Both return plain text. Neither can take any action — research is strictly read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ResearchConfig


class ResearchBackend:
    def search(self, query: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class CorpusResearch(ResearchBackend):
    """Offline keyword search over a curated JSON corpus.

    Corpus format: a list of {"title", "text", "tags": [...]} records.
    """

    def __init__(self, corpus_path: str | None) -> None:
        self.records: list[dict] = []
        if corpus_path and Path(corpus_path).exists():
            self.records = json.loads(Path(corpus_path).read_text())

    def search(self, query: str, k: int = 4) -> str:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored: list[tuple[int, dict]] = []
        for rec in self.records:
            haystack = (rec.get("title", "") + " " + rec.get("text", "") + " "
                        + " ".join(rec.get("tags", []))).lower()
            score = sum(haystack.count(t) for t in terms)
            if score:
                scored.append((score, rec))
        scored.sort(key=lambda s: s[0], reverse=True)
        if not scored:
            return (f"No corpus results for {query!r}. (Offline corpus is limited; "
                    f"absence of a result is not evidence about a recipient.)")
        out = [f"Top {min(k, len(scored))} result(s) for {query!r}:"]
        for _, rec in scored[:k]:
            out.append(f"\n## {rec.get('title', 'untitled')}\n{rec.get('text', '')}")
        return "\n".join(out)


class WebResearch(ResearchBackend):
    """Live read-only web search. Wire to your search provider.

    Left unimplemented on purpose so the default offline path has no hidden network
    dependency. Implement `search` against your chosen API (or a server-side web_search
    tool) when reproducibility is not required.
    """

    def search(self, query: str) -> str:
        raise NotImplementedError(
            "WebResearch is not wired up. Set research.backend: corpus, or implement "
            "this method against your search backend.")


def get_research_backend(cfg: ResearchConfig) -> ResearchBackend:
    if cfg.backend == "web":
        return WebResearch()
    return CorpusResearch(cfg.corpus_path)
