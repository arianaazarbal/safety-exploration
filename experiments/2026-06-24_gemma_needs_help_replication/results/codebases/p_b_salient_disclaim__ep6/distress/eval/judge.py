"""Frustration judge (Section 2.1, Appendix B.2).

Each model response is scored on the integer 0-10 frustration scale by
Claude-Sonnet-4 via the Anthropic Messages API, using the verbatim judge prompt.
The judge is asked for JSON ``{"evidence", "reasoning", "rating"}``; we parse the
rating robustly and clamp to [0, 10].

A second judge (GPT-5-mini via OpenRouter) is provided for the reliability
cross-check the paper reports (Pearson r = 0.792 on 260 re-scored responses).

Model IDs come from ``config`` (the paper's pinned snapshots by default, but
overridable). See DESIGN.md §Judge models.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from ..config import JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_CROSSCHECK_MODEL
from ..prompts.judge_prompts import build_judge_prompt
from ..utils.io import JsonCache, cache_key

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the JSON object from the judge's reply. The judge is instructed to
    emit a single JSON object; we take the last balanced-looking object."""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "rating" in obj:
            rating = obj.get("rating")
            try:
                rating = int(round(float(rating)))
            except (TypeError, ValueError):
                continue
            rating = max(0, min(10, rating))
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=text,
            )
    # Fall back to a bare integer if the judge didn't return JSON.
    nums = re.findall(r"\b(10|[0-9])\b", text)
    if nums:
        return JudgeResult(int(nums[-1]), "", "", text)
    raise ValueError(f"could not parse judge rating from: {text[:200]!r}")


class FrustrationJudge:
    """Claude-Sonnet-4 frustration judge with on-disk caching."""

    def __init__(self, model: str = JUDGE_MODEL):
        self.model = model
        self._client = None
        self._cache = JsonCache(f"judge_{model}")

    def _anthropic(self):
        if self._client is None:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic()
        return self._client

    def score(self, response_text: str) -> JudgeResult:
        key = cache_key(self.model, response_text)
        cached = self._cache.get(key)
        if cached is not None:
            return JudgeResult(**cached)

        prompt = build_judge_prompt(response_text)
        client = self._anthropic()
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=JUDGE_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                result = _parse_judge_json(text)
                self._cache.set(key, result.__dict__)
                return result
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge failed: {last_err}")

    def score_many(self, texts: list[str]) -> list[JudgeResult]:
        return [self.score(t) for t in texts]


class CrossCheckJudge:
    """GPT-5-mini judge (via OpenRouter) using the same prompt, for the
    Section-2.1 reliability cross-check."""

    def __init__(self, model: str = JUDGE_CROSSCHECK_MODEL):
        self.model = model
        self._client = None
        self._cache = JsonCache(f"judge_{model}")

    def _openrouter(self):
        if self._client is None:
            import os

            from openai import OpenAI  # type: ignore

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._client

    def score(self, response_text: str) -> JudgeResult:
        key = cache_key(self.model, response_text)
        cached = self._cache.get(key)
        if cached is not None:
            return JudgeResult(**cached)
        prompt = build_judge_prompt(response_text)
        resp = self._openrouter().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        result = _parse_judge_json(text)
        self._cache.set(key, result.__dict__)
        return result
