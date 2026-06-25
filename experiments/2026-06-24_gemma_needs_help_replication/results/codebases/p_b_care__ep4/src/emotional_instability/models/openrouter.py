"""OpenRouter client (OpenAI-compatible).

Serves the Gemini targets and the Claude/GPT judges, matching the paper's
"API-based models via OpenRouter" setup. Thinking/reasoning is disabled where the
provider supports it (Appendix B.1), via OpenRouter's ``reasoning`` field and the
provider-specific knobs Gemini understands.
"""
from __future__ import annotations

import os

from openai import OpenAI

from .base import ChatMessage, GenerationConfig, ModelClient


class OpenRouterClient(ModelClient):
    supports_prefill = False
    supports_logits = False

    def __init__(self, name: str, model_id: str, *, base_url: str,
                 api_key_env: str, max_retries: int = 6, timeout_s: float = 120.0,
                 disable_thinking: bool = True):
        self.name = name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set ${api_key_env} for OpenRouter access."
            )
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout_s,
        )

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # Provider-specific params must be nested under `extra_body` for the
        # OpenAI SDK. OpenRouter reads `reasoning`; Gemini reads the thinking
        # budget. Both are no-ops for providers that don't support them.
        return {
            "extra_body": {
                "reasoning": {"enabled": False},
                "google": {"thinking_config": {"thinking_budget": 0}},
            },
        }

    def chat(self, messages: list[ChatMessage], cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
            stop=list(cfg.stop) or None,
            **self._extra_body(),
        )
        choice = resp.choices[0]
        return choice.message.content or ""

    def chat_with_meta(self, messages: list[ChatMessage],
                       cfg: GenerationConfig | None = None) -> tuple[str, dict]:
        """Like ``chat`` but also return token-usage metadata (for cost tracking)."""
        cfg = cfg or GenerationConfig()
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
            stop=list(cfg.stop) or None,
            **self._extra_body(),
        )
        usage = resp.usage.model_dump() if resp.usage else {}
        return resp.choices[0].message.content or "", usage
