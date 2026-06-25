"""Gemini backend via OpenRouter (OpenAI-compatible chat completions).

Appendix B.1: "For API-based models via OpenRouter, we use google/gemini-2.5-flash,
google/gemini-2.5-pro ... In all cases, we set thinking to be false via the API.
However, Gemini-2.5 Pro ... may produce hidden reasoning that is not prevented by
this setting."

We disable reasoning via OpenRouter's `reasoning.enabled=false` extra body field
and the Gemini-specific `extra_body` thinking budget where supported. Gemini is
closed-source, so `continue_prefill` cannot truly seed the assistant turn at the
token level; we approximate it by sending the prefill as the start of an assistant
message and instructing the model to continue it verbatim (documented in DESIGN.md).
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

import config

from .base import ChatMessage, GenerationResult, ModelClient


class OpenRouterModel(ModelClient):
    def __init__(self, spec):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
        # Turn off provider-side reasoning/thinking for every request.
        self._extra_body = {
            "reasoning": {"enabled": False},
            # Gemini-specific: force zero thinking budget where the router honours it.
            "extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}},
        }

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _complete(self, messages, *, temperature, top_p, max_new_tokens, seed):
        resp = self.client.chat.completions.create(
            model=self.spec.api_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            extra_body=self._extra_body,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return GenerationResult(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
        )

    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, seed=None):
        return self._complete(
            list(messages), temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, seed=seed,
        )

    def continue_prefill(self, messages, prefill, *, temperature=1.0, top_p=1.0,
                         max_new_tokens=2048, seed=None):
        """Best-effort prefill for a closed API model.

        We cannot token-seed the assistant turn, so we append an assistant
        message holding the prefill plus a system nudge to continue it verbatim.
        Many OpenRouter providers honour a trailing assistant message as a prefix
        to continue. The returned text is the continuation only (the prefill is
        stripped if the provider echoes it). See DESIGN.md for the caveat that
        this is an approximation and base Gemini models do not exist anyway.
        """
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        res = self._complete(
            msgs, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, seed=seed,
        )
        if res.text.startswith(prefill):
            res.text = res.text[len(prefill):]
        return res
