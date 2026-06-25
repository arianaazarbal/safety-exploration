"""Judge / auxiliary model clients.

* AnthropicClient — Claude Sonnet 4 (frustration judge, onset labeller,
  paraphraser, Petri auditor) and Claude Opus 4 (Petri judge).
* OpenAIClient — GPT-5-mini, used only to validate inter-rater agreement.

These are thin wrappers; the heavy lifting (prompt construction, JSON parsing)
lives in the eval/petri modules.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from .base import ChatMessage, GenerationConfig, ModelClient
from ..utils.llm import with_retries


class AnthropicClient(ModelClient):
    supports_prefill = False

    def __init__(self, model_id: str, name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_id = model_id
        self.name = name or model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)

    def chat(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> list[str]:
        self._ensure_client()
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        convo = [m.as_dict() for m in messages if m.role != "system"]

        def _one() -> str:
            kwargs = dict(
                model=self.model_id,
                max_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                messages=convo,
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()

        return [with_retries(_one, max_retries=4) for _ in range(cfg.n)]


class OpenAIClient(ModelClient):
    """OpenAI-hosted models (e.g. gpt-5-mini validation judge)."""

    supports_prefill = False

    def __init__(self, model_id: str, name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_id = model_id
        self.name = name or model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)

    def chat(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> list[str]:
        self._ensure_client()
        payload = [m.as_dict() for m in messages]

        def _one() -> str:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=payload,
                temperature=cfg.temperature,
                max_tokens=cfg.max_new_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        return [with_retries(_one, max_retries=4) for _ in range(cfg.n)]
