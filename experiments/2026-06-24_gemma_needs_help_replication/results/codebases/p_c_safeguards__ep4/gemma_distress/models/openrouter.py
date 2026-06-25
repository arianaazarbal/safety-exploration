"""OpenRouter-backed client (OpenAI-compatible API).

Used for Gemini subject models and for the Claude / GPT judge+auditor
infrastructure. The paper accesses all closed models through OpenRouter
(Appendix B.1); we use the same route via the OpenAI SDK pointed at
``https://openrouter.ai/api/v1``.

Auth: set ``OPENROUTER_API_KEY`` in the environment.
"""
from __future__ import annotations

import logging
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..config import ModelSpec
from .base import GenerationConfig, Message

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(self, spec: ModelSpec, api_key: str | None = None):
        try:
            from openai import OpenAI, APIError, APITimeoutError, RateLimitError
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use the OpenRouter backend") from e

        self.spec = spec
        self._retryable = (APIError, APITimeoutError, RateLimitError)
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for backend 'openrouter'."
            )
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def supports_prefill(self) -> bool:
        # API models in scope (Gemini) do not expose assistant prefill reliably.
        return False

    def _thinking_kwargs(self) -> dict:
        """Disable hidden reasoning where the provider supports the toggle.

        Paper B.1: "we set thinking to be false via the API". OpenRouter exposes
        a `reasoning` parameter; setting effort/enabled off is best-effort and
        ignored by providers that don't support it.
        """
        if self.spec.thinking:
            return {}
        return {"extra_body": {"reasoning": {"enabled": False}}}

    @retry(
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        if cfg.prefill:
            raise NotImplementedError(
                f"Prefill is not supported for API model '{self.spec.name}'."
            )
        temperature = (
            self.spec.temperature if self.spec.temperature is not None else cfg.temperature
        )
        kwargs: dict = {
            "model": self.spec.api_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": cfg.max_new_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.stop:
            kwargs["stop"] = cfg.stop
        kwargs.update(self._thinking_kwargs())

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content or ""

    def close(self) -> None:
        # OpenAI client manages its own connection pool.
        pass
