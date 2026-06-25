"""Gemini target models via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter (google/gemini-2.5-flash,
google/gemini-2.5-pro) with thinking disabled. OpenRouter exposes an
OpenAI-compatible Chat Completions API, so we use the ``openai`` SDK pointed at
the OpenRouter base URL.

Gemini is API-only here: no prefilling and no hidden states, exactly as the
paper notes for closed models. ``thinking=False`` is requested, with the caveat
(from the paper) that Gemini-2.5-Pro may still produce hidden reasoning.
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from emotional_stability.config import GEMINI_API_MODELS, Settings
from emotional_stability.models.base import ChatModel, GenerationConfig
from emotional_stability.records import Message


class GeminiOpenRouterModel(ChatModel):
    supports_prefill = False
    supports_hidden_states = False

    def __init__(self, name: str, settings: Settings | None = None):
        if name not in GEMINI_API_MODELS:
            raise ValueError(f"unknown Gemini model key: {name}")
        self.name = name
        self.openrouter_id = GEMINI_API_MODELS[name]
        self.settings = (settings or Settings.load()).require("openrouter_api_key")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
        )

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        extra_body: dict = {}
        if not cfg.thinking:
            # OpenRouter reasoning control; Gemini maps this to thinking budget.
            extra_body["reasoning"] = {"enabled": False}
        resp = self._client.chat.completions.create(
            model=self.openrouter_id,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""
