"""Thin wrappers around the Anthropic and OpenAI SDKs for the judge / auditor /
paraphrase calls. These are *infrastructure* models (Claude, GPT), distinct from
the Gemma/Gemini targets under evaluation.

Model IDs are pinned to the exact strings the paper reports (e.g.
``claude-sonnet-4-20250514``) for faithful replication; swap them in config if
those snapshots are retired.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


# --------------------------------------------------------------------------- #
# Anthropic (Claude judges / auditors)
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, model: str, *, max_tokens: int = 1024, temperature: float = 0.0):
        from anthropic import Anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic(api_key=key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60))
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60))
    def chat(self, messages: list[dict], *, system: str | None = None) -> str:
        """Multi-turn variant for the Petri auditor loop."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- #
# OpenAI (GPT-5-mini judge-validation)
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(self, model: str, *, max_tokens: int = 1024, temperature: float = 0.0):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(api_key=key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60))
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# JSON extraction shared by judge / onset / paraphrase parsers
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the last JSON object out of a model response.

    Judges are instructed to emit JSON, sometimes after free-text reasoning; we
    grab the outermost ``{...}`` and tolerate the curly/smart quotes that appear
    in the paper's prompt examples.
    """
    cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = list(_JSON_RE.finditer(cleaned))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON in response:\n{text[:500]}")
