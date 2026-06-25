"""OpenRouter backend for the Gemini models (Appendix B.1).

The paper accesses ``google/gemini-2.5-flash`` and ``google/gemini-2.5-pro``
through OpenRouter with thinking disabled. OpenRouter exposes an
OpenAI-compatible ``/chat/completions`` endpoint; we call it directly with
``requests`` and a bounded exponential backoff so long sampling runs survive
transient 429/5xx responses.

Note (matching the paper): disabling reasoning is best-effort. Gemini-2.5-Pro in
particular "may produce hidden reasoning that is not prevented by this setting"
(Appendix B.1).
"""

from __future__ import annotations

import random
import time
from typing import Sequence

from ..config import ModelSpec, SamplingConfig, require_env
from ..logging_utils import get_logger
from .base import ChatMessage, GenerationResult, ModelClient

logger = get_logger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.name = spec.name
        self.model_id = spec.model_id
        self._api_key = require_env("OPENROUTER_API_KEY")
        self._options = spec.options or {}
        self.max_concurrency = int(self._options.get("max_concurrency", 8))

    def chat_batch(self, conversations, sampling):  # type: ignore[override]
        from ..concurrency import concurrent_map

        return concurrent_map(
            lambda conv: self.chat(conv, sampling),
            list(conversations),
            self.max_concurrency,
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers OpenRouter recommends.
            "HTTP-Referer": "https://github.com/replication/gemma-needs-help",
            "X-Title": "Gemma Needs Help replication",
        }

    def _payload(
        self, messages: Sequence[ChatMessage], sampling: SamplingConfig
    ) -> dict:
        payload: dict = {
            "model": self.model_id,
            "messages": list(messages),
            "temperature": sampling.temperature,
            "max_tokens": sampling.max_new_tokens,
        }
        if sampling.top_p < 1.0:
            payload["top_p"] = sampling.top_p
        if sampling.top_k > 0:
            payload["top_k"] = sampling.top_k
        # Disable hidden reasoning where the option requests it.
        if self._options.get("reasoning_enabled") is False:
            payload["reasoning"] = {"enabled": False}
        return payload

    def chat(
        self, messages: Sequence[ChatMessage], sampling: SamplingConfig
    ) -> GenerationResult:
        import requests

        payload = self._payload(messages, sampling)
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                resp = requests.post(
                    _OPENROUTER_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=180,
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                return GenerationResult(
                    text=text.strip(),
                    finish_reason=choice.get("finish_reason"),
                    raw=data.get("usage"),
                )
            except _Retryable as exc:
                last_err = exc
                delay = min(2**attempt + random.uniform(0, 1), 60)
                logger.warning("OpenRouter retry %d after %.1fs: %s", attempt + 1, delay, exc)
                time.sleep(delay)
            except requests.RequestException as exc:  # network errors → retry
                last_err = exc
                delay = min(2**attempt + random.uniform(0, 1), 60)
                logger.warning("OpenRouter network retry %d after %.1fs", attempt + 1, delay)
                time.sleep(delay)
        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")


class _Retryable(Exception):
    pass
