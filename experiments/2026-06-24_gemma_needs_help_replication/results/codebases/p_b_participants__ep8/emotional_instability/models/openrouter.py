"""OpenRouter (OpenAI-compatible) backend for Gemini and the GPT-5-mini judge.

The paper accesses Gemini-2.5-{flash,pro} via OpenRouter and disables thinking
(Appendix B.1). OpenRouter exposes an OpenAI-compatible Chat Completions API, so
we use the ``openai`` SDK pointed at the OpenRouter base URL.

Prefilling on Gemini: closed Gemini cannot be prefilled or fine-tuned (the paper
notes this limitation explicitly, Section 6). ``continue_prefill`` therefore
falls back to an assistant-prefix message, which OpenRouter passes through for
providers that support it; for Gemini specifically the prefill experiment is out
of scope (no base model, no true prefill), so this path is only used defensively.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, Generation


class OpenRouterClient:
    def __init__(
        self,
        model_id: str,
        spec_name: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        disable_thinking: bool = True,
    ) -> None:
        from openai import OpenAI

        self.spec_name = spec_name
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "MISSING_OPENROUTER_API_KEY"),
        )

    def _extra_body(self) -> dict:
        # OpenRouter "reasoning" knob: exclude/disable thinking per Appendix B.1.
        # Gemini-2.5-pro may still emit hidden reasoning the API cannot suppress.
        if self.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60))
    def _chat(self, messages: list[dict], temperature: float,
              max_new_tokens: int, seed: Optional[int]) -> Generation:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Generation(
            text=(choice.message.content or "").strip(),
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def generate(self, messages, *, temperature=1.0, max_new_tokens=2048,
                 seed=None) -> Generation:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self._chat(payload, temperature, max_new_tokens, seed)

    def continue_prefill(self, messages, prefill, *, temperature=1.0,
                         max_new_tokens=2048, seed=None) -> Generation:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        payload.append({"role": "assistant", "content": prefill})
        gen = self._chat(payload, temperature, max_new_tokens, seed)
        # Some providers echo the prefill; strip it defensively.
        text = gen.text
        if text.startswith(prefill):
            text = text[len(prefill):]
        return Generation(text=text, finish_reason=gen.finish_reason)
