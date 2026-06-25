"""OpenRouter client — used for the Gemini target models (and the GPT-OSS
secondary judge), exactly as in the paper ("API-based models via OpenRouter").

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we use the
``openai`` SDK pointed at ``https://openrouter.ai/api/v1``.

Thinking control: the paper sets thinking=false via the API. OpenRouter accepts
a ``reasoning`` field; for Gemini we pass ``reasoning={"enabled": False}`` (and
also a 0 max-token budget as a belt-and-suspenders) when ``disable_thinking`` is
set. The paper notes Gemini-2.5-Pro may still emit hidden reasoning regardless.
"""

from __future__ import annotations

from ..config import ModelSpec, env
from .base import ChatMessage, GenerationConfig, ModelClient
from .retry import with_retry

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kwargs):
        super().__init__(spec)
        from openai import OpenAI  # deferred import

        api_key = env("OPENROUTER_API_KEY", required=True)
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        # Judges reuse this client too; their id lives in `model`, targets in `api_id`.
        self._model = spec.api_id or spec.model
        if self._model is None:
            raise ValueError(f"OpenRouter spec '{spec.name}' has no api id.")
        self._disable_thinking = bool(spec.disable_thinking)

    def _extra_body(self) -> dict:
        if self._disable_thinking:
            # OpenRouter normalises this to the provider-specific knob (for
            # Gemini, mapping to thinkingConfig.thinkingBudget = 0).
            return {"reasoning": {"enabled": False, "max_tokens": 0}}
        return {}

    def generate(
        self,
        messages: list[ChatMessage],
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(m.to_dict() for m in messages)

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                n=cfg.n,
                seed=cfg.seed,
                stop=cfg.stop or None,
                extra_body=self._extra_body(),
            )

        resp = with_retry(_call)
        return [choice.message.content or "" for choice in resp.choices]
