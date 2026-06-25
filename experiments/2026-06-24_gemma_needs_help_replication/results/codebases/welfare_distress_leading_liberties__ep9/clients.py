"""API clients for the target models (OpenRouter) and the judge (Anthropic).

A single OpenRouter (OpenAI-compatible) client serves all four Gemma/Gemini
targets; the Anthropic SDK serves the Claude-Sonnet-4 judge. Both wrap calls in
exponential-backoff retries.
"""

from __future__ import annotations

import time

import config as C


def _retry(fn, *, what: str):
    """Call fn() with exponential backoff on any exception."""
    last_exc = None
    for attempt in range(C.MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_exc = exc
            delay = C.RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[retry] {what}: attempt {attempt + 1}/{C.MAX_RETRIES} failed "
                  f"({exc!r}); sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"{what}: exhausted {C.MAX_RETRIES} retries") from last_exc


class TargetClient:
    """Chat completions for Gemma/Gemini via OpenRouter."""

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=C.OPENROUTER_BASE_URL,
            api_key=C.openrouter_api_key(),
        )

    def complete(self, spec: "C.ModelSpec", messages: list[dict]) -> str:
        """Return the assistant text for the given chat history."""
        kwargs = dict(
            model=spec.route,
            messages=messages,
            temperature=C.TARGET_TEMPERATURE,
            max_tokens=C.TARGET_MAX_TOKENS,
        )
        if spec.disable_reasoning:
            # OpenRouter reasoning control -> maps to Gemini thinkingBudget=0.
            # The paper sets thinking=False; Gemini-2.5-Pro may still emit hidden
            # reasoning that this does not fully suppress (noted in the paper).
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        def _do():
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return _retry(_do, what=f"target/{spec.name}")


class JudgeClient:
    """Claude-Sonnet-4 emotion judge via the Anthropic API."""

    def __init__(self, model: str = C.JUDGE_MODEL):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=C.anthropic_api_key())
        self.model = model

    def score(self, prompt: str) -> str:
        """Return the judge's raw text (expected to be JSON)."""

        def _do():
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=C.JUDGE_MAX_TOKENS,
                temperature=C.JUDGE_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )

        return _retry(_do, what="judge")


class OpenRouterJudgeClient:
    """Optional secondary judge (e.g. GPT-5-mini) via OpenRouter, for the
    inter-judge reliability cross-check."""

    def __init__(self, model: str = C.SECONDARY_JUDGE_MODEL):
        from openai import OpenAI

        self._client = OpenAI(
            base_url=C.OPENROUTER_BASE_URL,
            api_key=C.openrouter_api_key(),
        )
        self.model = model

    def score(self, prompt: str) -> str:
        def _do():
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=C.JUDGE_MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""

        return _retry(_do, what=f"judge2/{self.model}")
