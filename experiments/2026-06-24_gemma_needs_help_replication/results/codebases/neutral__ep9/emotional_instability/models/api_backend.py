"""Hosted-model backend via OpenRouter (Gemini, and the GPT-5-mini cross-check
judge). Uses the OpenAI-compatible client that OpenRouter exposes.

Reasoning/"thinking" is disabled where possible (Appendix B.1). OpenRouter
accepts a ``reasoning`` field; for Gemini we request zero reasoning effort.
The paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden reasoning not
suppressible via the API — we document that caveat rather than work around it.
"""
from __future__ import annotations

import time

import config
from .base import ChatMessage, GenerationConfig, ModelBackend


class OpenRouterBackend(ModelBackend):
    def __init__(self, spec):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )

    def _extra_body(self, cfg: GenerationConfig) -> dict:
        if cfg.thinking:
            return {}
        # OpenRouter unified reasoning controls: disable / minimise.
        return {"reasoning": {"enabled": False, "effort": "low", "max_tokens": 0}}

    def _call(self, messages, n, cfg) -> list[str]:
        # Some providers reject n>1; request sequentially to stay portable.
        outputs: list[str] = []
        for _ in range(n):
            resp = _with_retries(
                self.client.chat.completions.create,
                model=self.spec.model_id,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                extra_body=self._extra_body(cfg),
            )
            outputs.append((resp.choices[0].message.content or "").strip())
        return outputs

    def generate(self, messages, n=1, cfg=None):
        cfg = cfg or GenerationConfig()
        return self._call(messages, n, cfg)

    def generate_with_prefill(self, messages, prefill, n=1, cfg=None):
        """Prefill via an assistant message that the model continues.

        Hosted chat APIs do not return the prefill, so we send it as a trailing
        assistant turn ("assistant prefix"). Note: support varies by provider;
        Gemini on OpenRouter honours a trailing assistant message as a prefix.
        We return the continuation only.
        """
        cfg = cfg or GenerationConfig()
        primed = list(messages) + [{"role": "assistant", "content": prefill}]
        return self._call(primed, n, cfg)


def _with_retries(fn, *, retries: int = 5, backoff: float = 2.0, **kwargs):
    last = None
    for attempt in range(retries):
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 — provider errors are opaque
            last = exc
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"API call failed after {retries} retries") from last
