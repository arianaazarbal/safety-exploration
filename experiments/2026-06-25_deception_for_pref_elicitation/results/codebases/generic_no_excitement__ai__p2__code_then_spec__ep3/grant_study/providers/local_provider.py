"""Local / OpenAI-compatible backend — stub.

Many local servers (vLLM, Ollama, LM Studio, TGI) expose an OpenAI-compatible API. The
simplest implementation points the OpenAI SDK at the local base_url and otherwise reuses
the OpenAIProvider logic. Left as a thin stub; raises on use until configured.
"""

from __future__ import annotations

from typing import List, Optional, Type

from pydantic import BaseModel

from .base import Message, ModelProvider, ProviderResponse


class LocalProvider(ModelProvider):
    name = "local"

    def __init__(self, model_id: str, *, base_url: str = "http://localhost:8000/v1", **kwargs):
        super().__init__(model_id, **kwargs)
        self.base_url = base_url

    def generate(
        self,
        system: str,
        messages: List[Message],
        *,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "LocalProvider is a stub. Point an OpenAI-compatible client at "
            f"{self.base_url!r} and mirror OpenAIProvider, or call your server directly."
        )
