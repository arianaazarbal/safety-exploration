"""OpenRouter backend for Gemini (and the GPT-5-mini judge-agreement model).

Appendix B.1 accesses the API models through OpenRouter, with "thinking" set to false. We
use the OpenAI-compatible client pointed at OpenRouter's endpoint. Reasoning is disabled
via OpenRouter's unified ``reasoning`` parameter when ``thinking`` is false; the paper
notes Gemini-2.5-Pro and GPT-5.x may still produce hidden reasoning the flag cannot
suppress, so this is best-effort and documented.

``n`` samples are issued as independent concurrent requests rather than relying on the
provider's ``n`` parameter, which is inconsistently supported across OpenRouter providers.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..utils import env, parallel_map, with_retries
from .base import ChatModel, Conversation

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ChatModel):
    """API model accessed via OpenRouter's OpenAI-compatible endpoint."""

    supports_prefill = False  # closed API models do not expose true assistant prefilling

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        thinking: bool = False,
        max_workers: int = 8,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = _OPENROUTER_BASE_URL,
    ):
        super().__init__(name)
        from openai import OpenAI

        self.model_id = model_id
        self.thinking = thinking
        self.max_workers = max_workers
        self._client = OpenAI(api_key=env(api_key_env, required=True), base_url=base_url)

    def _one(
        self, conversation: Conversation, *, temperature: float, max_new_tokens: int
    ) -> str:
        extra_body = {}
        if not self.thinking:
            # OpenRouter unified reasoning control: disable where the provider honours it.
            extra_body["reasoning"] = {"enabled": False}

        def call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=conversation,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            return content or ""

        return with_retries(call, label=f"openrouter:{self.model_id}")

    def chat_batch(
        self,
        conversations: list[Conversation],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        # Flatten into (conversation_index, sample_index) tasks for bounded concurrency.
        tasks = [(ci, conv) for ci, conv in enumerate(conversations) for _ in range(n)]
        outputs = parallel_map(
            lambda t: self._one(
                t[1], temperature=temperature, max_new_tokens=max_new_tokens
            ),
            tasks,
            max_workers=self.max_workers,
            desc=f"sample:{self.name}",
        )
        grouped: list[list[str]] = [[] for _ in conversations]
        for (ci, _), out in zip(tasks, outputs):
            grouped[ci].append(out)
        return grouped
