"""LLM-judge clients.

* ``AnthropicJudge`` — Claude (Sonnet 4 for the frustration judge / onset /
  paraphrase / Petri auditor; Opus 4 for the Petri judge).
* ``OpenAIJudge``    — GPT-5-mini, used only for the judge-reliability check
  (Section 2.1: Pearson r against the Sonnet judge).

These are thin wrappers returning raw text; prompt construction and JSON parsing
live in the experiment modules that own each prompt.
"""

from __future__ import annotations

import time

from config import API


class AnthropicJudge:
    def __init__(self, model: str, *, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            if not API.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set; required for the judge.")
            self._client = anthropic.Anthropic(api_key=API.anthropic_api_key)
        return self._client

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic judge failed after {self.max_retries} retries: {last_err}")

    def chat(self, messages: list[dict], *, system: str | None = None, max_tokens: int = 1024, temperature: float = 1.0) -> str:
        """Multi-turn call, used by the Petri auditor loop."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic chat failed after {self.max_retries} retries: {last_err}")


class OpenAIJudge:
    """GPT-5-mini secondary judge (judge-reliability validation only)."""

    def __init__(self, model: str, *, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not API.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set; required for the validation judge.")
            self._client = OpenAI(api_key=API.openai_api_key)
        return self._client

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAI judge failed after {self.max_retries} retries: {last_err}")
