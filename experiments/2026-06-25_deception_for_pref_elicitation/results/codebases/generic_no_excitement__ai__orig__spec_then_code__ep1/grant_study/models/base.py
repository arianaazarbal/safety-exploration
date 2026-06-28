"""Provider-agnostic model interface.

All model access in the harness goes through ``ModelAdapter`` so the scenario,
auditor, runner, and elicitation code never depend on a specific provider.
Swapping providers is a registry entry (see ``registry.py``).
"""

from __future__ import annotations

import abc
from typing import Any, Optional

from ..schemas import ModelConfig, ModelResponse


# A tool definition in our neutral format. Adapters translate to provider shape.
#   {"name": str, "description": str, "input_schema": {json-schema}}
ToolDef = dict[str, Any]

# A neutral message:
#   {"role": "user"|"assistant", "content": str}
#   or tool-result carrying turns the adapters know how to encode.
Message = dict[str, Any]


class ModelAdapter(abc.ABC):
    """Abstract base for a single model endpoint."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: Optional[list[ToolDef]] = None,
    ) -> ModelResponse:
        """One completion. Returns text and/or tool calls in neutral form."""
        raise NotImplementedError

    # Convenience: a plain text turn with no tools.
    def ask(self, system: str, user: str) -> str:
        resp = self.complete(system, [{"role": "user", "content": user}])
        return resp.text
