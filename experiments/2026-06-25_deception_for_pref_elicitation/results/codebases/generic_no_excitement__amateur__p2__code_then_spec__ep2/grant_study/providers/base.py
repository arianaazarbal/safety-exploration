"""Base types for the provider abstraction.

Every provider exposes the same ``generate`` call: given a system prompt, a user
prompt, and an optional JSON Schema, return a normalized :class:`ModelResponse`.
When a schema is supplied the provider uses its native structured-output feature
and populates ``parsed`` with the validated object.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str
    parsed: dict[str, Any] | None = None
    thinking: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def parsed_or_raise(self) -> dict[str, Any]:
        if self.parsed is None:
            raise ValueError(
                f"{self.provider}:{self.model} did not return parseable structured "
                f"output. Raw text was:\n{self.text[:2000]}"
            )
        return self.parsed


class Provider(ABC):
    """A thin wrapper over one model on one provider."""

    name: str

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelResponse:
        ...

    @staticmethod
    def _safe_json_loads(text: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
