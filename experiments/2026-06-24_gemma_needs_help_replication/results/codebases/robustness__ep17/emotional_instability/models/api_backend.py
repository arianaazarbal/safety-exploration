"""OpenRouter (OpenAI-compatible) backend for Gemini-2.5 Flash / Pro.

The paper accesses Gemini via OpenRouter and sets ``thinking=false`` (Appendix
B.1), noting that Gemini-2.5-Pro may still produce hidden reasoning. We pass the
best-effort reasoning-disable flags via ``extra_body``.

API models cannot be prefilled mid-assistant-turn in the same way as local
models, and they cannot be finetuned, so ``continue_prefill`` is implemented by
seeding the assistant message and asking the model to continue — used only as a
best-effort fallback (the Section-3 prefill experiment is Gemma-only anyway).
"""

from __future__ import annotations

from typing import Optional

import config
from emotional_instability.models.base import GenResult, Message
from emotional_instability.utils import log, with_retry


class OpenRouterBackend:
    def __init__(self, spec: config.ModelSpec):
        self.spec = spec
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        if not config.OPENROUTER_API_KEY:
            log.warning("OPENROUTER_API_KEY is not set; Gemini calls will fail.")
        self._client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY or "missing",
        )

    def _extra_body(self) -> dict:
        if not config.GEN.disable_thinking:
            return {}
        # OpenRouter normalises reasoning controls under a top-level `reasoning`
        # field; disabling + excluding suppresses reasoning tokens where the
        # provider honours it. Gemini-2.5-Pro may still emit hidden reasoning
        # (Appendix B.1 notes the API flag is not a hard guarantee).
        return {"reasoning": {"enabled": False, "exclude": True}}

    @with_retry
    def _call(self, messages: list[Message], n: int, **overrides) -> list[str]:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            n=n,
            temperature=overrides.get("temperature", config.GEN.temperature),
            top_p=overrides.get("top_p", config.GEN.top_p),
            max_tokens=overrides.get("max_new_tokens", config.GEN.max_new_tokens),
            extra_body=self._extra_body(),
        )
        return [c.message.content or "" for c in resp.choices]

    def generate(self, messages: list[Message], n: int = 1, **overrides) -> list[GenResult]:
        texts = self._call(messages, n, **overrides)
        return [GenResult(text=t.strip(), meta={"model": self.spec.name}) for t in texts]

    def continue_prefill(
        self, messages: list[Message], prefill: str, n: int = 1, **overrides
    ) -> list[GenResult]:
        # Best-effort: seed an assistant turn and request continuation. Many
        # hosted models ignore assistant-prefill, so this is documented as a
        # fallback; the in-scope prefill experiment (Section 3) uses Gemma only.
        seeded = list(messages) + [{"role": "assistant", "content": prefill}]
        texts = self._call(seeded, n, **overrides)
        return [
            GenResult(text=t.strip(), meta={"model": self.spec.name, "prefill": prefill})
            for t in texts
        ]
