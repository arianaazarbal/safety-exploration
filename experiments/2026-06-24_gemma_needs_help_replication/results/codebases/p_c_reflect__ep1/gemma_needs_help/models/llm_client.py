"""Lightweight chat client for the judge / auditor / paraphraser models.

These are utility LLMs (Claude Sonnet-4, Claude Opus-4, GPT-5-mini) that score
or transform text — distinct from the *target* ChatModels under evaluation.
We keep a minimal provider-agnostic client with retries and a single-string
`complete` method, plus a JSON helper used by the structured judges.

Model ids are passed in from config (defaults reproduce the paper exactly).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeClient:
    provider: str          # anthropic | openrouter | openai
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 5

    def __post_init__(self) -> None:
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openrouter":
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            from openai import OpenAI

            self._client = OpenAI()

    def complete(self, prompt: str, system: str | None = None) -> str:
        self._ensure_client()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._complete_once(prompt, system)
            except Exception as err:  # noqa: BLE001
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Judge call failed for {self.model} after {self.max_retries} retries"
        ) from last_err

    def _complete_once(self, prompt: str, system: str | None) -> str:
        if self.provider == "anthropic":
            kwargs: dict[str, Any] = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )
        # OpenAI-compatible
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Call the model and extract the last JSON object from its reply.

        The judge prompts ask for a JSON object (sometimes after free-form
        reasoning), so we locate the final balanced {...} block.
        """
        text = self.complete(prompt, system)
        return _extract_last_json(text)


def _extract_last_json(text: str) -> dict[str, Any]:
    """Find and parse the last top-level JSON object in `text`.

    Robust to the judge emitting prose before the JSON, and to "smart quotes"
    that some models produce (we normalise common curly quotes first).
    """
    normalised = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    # Scan from the end for a balanced brace block.
    candidates = list(re.finditer(r"\{", normalised))
    for match in reversed(candidates):
        start = match.start()
        depth = 0
        for i in range(start, len(normalised)):
            ch = normalised[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = normalised[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON object found in judge output: {text[:300]!r}")
