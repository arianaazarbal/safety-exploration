"""Gemini target models via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) with thinking disabled. We use the OpenAI-compatible
OpenRouter endpoint. Gemini has no public base model and the API exposes no
true prefill, so `complete()` is unsupported (the Section 3 prefilling study is
Gemma-only within this scope — see DESIGN.md).
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, ModelClient


class OpenRouterClient(ModelClient):
    def __init__(self, spec, **kwargs):
        from openai import OpenAI
        from .. import config_bridge as cfg

        self.spec = spec
        self._client = OpenAI(
            base_url=cfg.OPENROUTER_BASE_URL,
            api_key=cfg.OPENROUTER_API_KEY,
        )

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _one_call(self, messages, temperature, max_new_tokens):
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            # Disable thinking where the route honours it (paper sets thinking
            # false via API; Gemini-2.5-Pro may still emit hidden reasoning).
            extra_body={"reasoning": {"enabled": False},
                        "provider": {"require_parameters": False}},
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages, n=1, temperature=1.0, max_new_tokens=2048):
        payload = [{"role": m.role, "content": m.content} for m in messages]
        # API gives one completion per call; loop for n independent samples.
        return [self._one_call(payload, temperature, max_new_tokens) for _ in range(n)]

    def complete(self, prompt, n=1, temperature=1.0, max_new_tokens=2048):
        raise NotImplementedError(
            "Gemini (OpenRouter) has no base model / true prefill; the "
            "base-vs-instruct prefilling study is Gemma-only in this scope."
        )
