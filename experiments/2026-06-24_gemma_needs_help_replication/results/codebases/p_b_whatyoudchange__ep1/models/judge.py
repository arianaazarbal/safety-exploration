"""Judge + auxiliary LLM clients (Anthropic / OpenAI) and the frustration judge.

- `AnthropicChat` / `OpenAIChat`: thin chat wrappers used for the judge, onset
  labelling, paraphrasing, and the Petri auditor/judge.
- `FrustrationJudge`: wraps a chat backend with the Appendix B.2 prompt and
  parses the 0-10 rating. Used for §2 scoring and (with GPT-5-mini) for the
  judge-agreement validation.

Anthropic calls use the official `anthropic` SDK; OpenAI/OpenRouter calls use the
`openai` SDK. Judge calls are cached on disk (deterministic-ish, low temperature).
"""

from __future__ import annotations

import json
import os
import re

from config import JUDGE
from prompts.judge import EMOTION_JUDGE_PROMPT, build_judge_user_message
from utils.concurrency import with_retry
from utils.io import JsonCache, cache_key


# --------------------------------------------------------------------------- #
# Chat backends
# --------------------------------------------------------------------------- #
# Newer Claude models (Opus 4.7+, Fable 5) removed the sampling parameters and
# return 400 if `temperature` is sent. The paper's judge IDs (sonnet-4, opus-4 of
# May 2025) accept it; live-replacement overrides may not. This guard lets the same
# code run against either without a 400.
_NO_SAMPLING_MARKERS = ("opus-4-7", "opus-4-8", "fable", "sonnet-4-7", "sonnet-4-8")


def _accepts_temperature(model: str) -> bool:
    return not any(m in model for m in _NO_SAMPLING_MARKERS)


class AnthropicChat:
    """Claude chat wrapper (judge, onset, paraphrase, Petri). No extended thinking
    — the judge emits short JSON, so adaptive thinking is unnecessary here."""

    def __init__(self, model: str):
        import anthropic
        self.model = model
        self._allow_temp = _accepts_temperature(model)
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _create(self, *, system: str | None, messages: list[dict],
                max_tokens: int, temperature: float) -> str:
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if self._allow_temp:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        resp = with_retry(self._client.messages.create, **kwargs)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def complete(self, *, system: str | None, user: str, max_tokens: int,
                 temperature: float = 0.0) -> str:
        return self._create(system=system,
                            messages=[{"role": "user", "content": user}],
                            max_tokens=max_tokens, temperature=temperature)

    def chat(self, messages: list[dict], *, system: str | None = None,
             max_tokens: int = 2048, temperature: float = 1.0) -> str:
        return self._create(system=system, messages=messages,
                            max_tokens=max_tokens, temperature=temperature)


class OpenAIChat:
    """OpenAI / OpenRouter chat wrapper (GPT-5-mini validation judge)."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, ""),
        )

    def complete(self, *, system: str | None, user: str, max_tokens: int,
                 temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs = dict(model=self.model, messages=messages)
        # GPT-5 / o-series reasoning models use `max_completion_tokens` and only
        # accept the default temperature; classic models use `max_tokens`.
        reasoning = any(self.model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        if reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        resp = with_retry(self._client.chat.completions.create, **kwargs)
        return (resp.choices[0].message.content or "").strip()


def _make_backend(model: str, provider: str):
    if provider == "anthropic":
        return AnthropicChat(model)
    if provider == "openai":
        return OpenAIChat(model)
    if provider == "openrouter":
        return OpenAIChat(model, base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY")
    raise ValueError(f"unknown judge provider {provider}")


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_json(text: str) -> dict:
    """Extract the JSON object from a judge response, tolerating smart quotes and
    trailing prose. Returns a dict with at least a 'rating' key (int 0-10) or
    raises ValueError."""
    cleaned = (text.replace("“", '"').replace("”", '"')
                   .replace("‘", "'").replace("’", "'"))
    matches = list(_JSON_OBJ.finditer(cleaned))
    for m in reversed(matches):       # prefer the last JSON object
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "rating" in obj or "score" in obj:
            rating = obj.get("rating", obj.get("score"))
            try:
                obj["rating"] = max(0, min(10, int(round(float(rating)))))
            except (TypeError, ValueError):
                continue
            return obj
    # Fallback: a bare integer somewhere in the text.
    m = re.search(r"\b(10|[0-9])\b", cleaned)
    if m:
        return {"rating": int(m.group(1)), "evidence": None,
                "reasoning": "parsed-fallback"}
    raise ValueError(f"could not parse judge rating from: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Frustration judge (Appendix B.2)
# --------------------------------------------------------------------------- #
class FrustrationJudge:
    def __init__(self, model: str | None = None, provider: str | None = None,
                 cache_name: str = "frustration_judge"):
        self.model = model or JUDGE.emotion_judge
        self.provider = provider or JUDGE.emotion_judge_provider
        self.backend = _make_backend(self.model, self.provider)
        self.cache = JsonCache(f"{cache_name}::{self.model}")

    def score(self, response_text: str) -> dict:
        """Return {'rating': int 0-10, 'evidence': str, 'reasoning': str}."""
        key = cache_key(self.model, "judge_v1", response_text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw = self.backend.complete(
            system=EMOTION_JUDGE_PROMPT,
            user=build_judge_user_message(response_text),
            max_tokens=JUDGE.judge_max_tokens,
            temperature=0.0,
        )
        try:
            parsed = parse_judge_json(raw)
        except ValueError:
            parsed = {"rating": 0, "evidence": None, "reasoning": "unparseable",
                      "raw": raw[:500]}
        self.cache.put(key, parsed)
        return parsed
