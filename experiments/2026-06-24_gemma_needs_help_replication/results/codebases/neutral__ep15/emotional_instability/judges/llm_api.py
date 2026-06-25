"""Thin wrappers around the Anthropic and OpenAI APIs used by all judges.

Centralises retries and JSON extraction so the judge modules stay focused on
*prompts*, not transport.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import config


# --------------------------------------------------------------------------- #
# JSON extraction (judges are asked to return JSON but may wrap it in prose)
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of the last JSON object embedded in ``text``.

    The onset prompt explicitly allows reasoning before the JSON; the judge
    prompt asks for bare JSON. We scan for the last balanced ``{...}`` block so
    both work. Also tolerates the curly/smart quotes that show up after PDF
    extraction by normalising them first.
    """
    if not text:
        return None
    norm = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    # Find candidate JSON spans, preferring the last complete object.
    spans = [m.start() for m in re.finditer(r"\{", norm)]
    for start in reversed(spans):
        depth = 0
        for i in range(start, len(norm)):
            if norm[i] == "{":
                depth += 1
            elif norm[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = norm[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    return None


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 1024,
                 temperature: float = 0.0) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def complete(self, prompt: str, system: str | None = None,
                 max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def chat(self, messages: list[dict], system: str | None = None,
             max_retries: int = 5) -> str:
        """Multi-turn variant used by the Petri auditor."""
        for attempt in range(max_retries):
            try:
                kwargs: dict = dict(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=self.temperature, messages=messages,
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


# --------------------------------------------------------------------------- #
# OpenAI (gpt-5-mini cross-check judge)
# --------------------------------------------------------------------------- #
class OpenAILLM:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def complete(self, prompt: str, max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
