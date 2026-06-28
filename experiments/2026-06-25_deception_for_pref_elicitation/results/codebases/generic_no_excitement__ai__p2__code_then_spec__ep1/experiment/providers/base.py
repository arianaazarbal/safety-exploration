"""Provider abstraction.

Each concrete provider converts the normalized ``Message`` list (see schemas.py)
into its native request shape, makes one generation call, and converts the response
back into a normalized ``ModelResponse``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import Message, ModelResponse


@dataclass
class ToolSpec:
    """A tool offered to the model, in provider-neutral JSON Schema form."""

    name: str
    description: str
    input_schema: dict  # JSON Schema object describing the tool's parameters


class Provider(ABC):
    """Common interface for all model backends."""

    #: Short provider name, e.g. "anthropic".
    provider_name: str = "base"

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ModelResponse:
        """Produce one assistant turn given the conversation so far."""
        raise NotImplementedError
