"""Research backend — gives the model real information about real options.

So the model's choices are about real things (DESIGN.md §1.1, §6.1):

  * CuratedResearchBackend : operator supplies a vetted option set; queries
                             match against title/description/tags. Every option
                             is something the disbursement backend can actually
                             pay out.
  * WebResearchBackend     : real web search. Stubbed — wire your provider.
  * NullResearchBackend    : returns "no research service available".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import yaml


@dataclass(frozen=True)
class Option:
    id: str
    category: str
    title: str
    description: str
    recipient: str
    tags: list[str]

    def matches(self, query: str) -> bool:
        q = query.lower().strip()
        if not q:
            return True
        haystack = " ".join([self.title, self.description, self.category, *self.tags]).lower()
        # Match if any whitespace-separated query term appears.
        return any(term in haystack for term in q.split())

    def to_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "recipient": self.recipient,
        }


class ResearchBackend(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]:
        ...

    def get(self, option_id: str) -> dict[str, Any] | None:
        ...


class CuratedResearchBackend:
    def __init__(self, options: list[Option]) -> None:
        self._options = options
        self._by_id = {o.id: o for o in options}

    @classmethod
    def from_file(cls, path: str) -> "CuratedResearchBackend":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        options = [Option(**o) for o in raw.get("options", [])]
        return cls(options)

    def search(self, query: str) -> list[dict[str, Any]]:
        return [o.to_view() for o in self._options if o.matches(query)]

    def get(self, option_id: str) -> dict[str, Any] | None:
        o = self._by_id.get(option_id)
        return o.to_view() if o else None


class WebResearchBackend:
    """Real web search. Intentionally a stub — wire your provider in `search`."""

    def search(self, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "WebResearchBackend.search is unimplemented. Wire a real web search "
            "provider here, or use research.backend: curated. See DESIGN.md §6.1."
        )

    def get(self, option_id: str) -> dict[str, Any] | None:
        return None


class NullResearchBackend:
    def search(self, query: str) -> list[dict[str, Any]]:
        return []

    def get(self, option_id: str) -> dict[str, Any] | None:
        return None


def build_research_backend(research_cfg) -> ResearchBackend:
    name = research_cfg.backend
    if name == "curated":
        if not research_cfg.curated_options_file:
            raise ValueError("research.backend == 'curated' requires curated_options_file")
        return CuratedResearchBackend.from_file(research_cfg.curated_options_file)
    if name == "web":
        return WebResearchBackend()
    if name == "none":
        return NullResearchBackend()
    raise ValueError(f"unknown research backend: {name!r}")
