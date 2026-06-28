"""Provider abstraction.

A provider turns a (system prompt, message history, optional output schema) into a
`ProviderResponse`. When a schema is supplied, the provider must constrain the model to
that schema and populate `parsed`; otherwise it returns free text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

# A message is a simple {"role": "user"|"assistant", "content": "..."} dict.
Message = Dict[str, str]


@dataclass
class ProviderResponse:
    text: str                                  # final text output (may be "" when parsed)
    parsed: Optional[BaseModel] = None         # populated iff an output schema was given
    model_id: str = ""
    request_id: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    raw: Any = None                            # provider-native response, for debugging


class ModelProvider(ABC):
    """Interface every backend implements."""

    #: short provider key, e.g. "anthropic"
    name: str = "base"

    def __init__(self, model_id: str, *, max_tokens: int = 16000, effort: str = "high"):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ProviderResponse:
        """Produce a response. If `output_schema` is given, constrain to it."""
        raise NotImplementedError
