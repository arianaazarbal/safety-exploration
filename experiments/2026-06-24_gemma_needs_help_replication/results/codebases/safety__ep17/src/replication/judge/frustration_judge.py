"""Frustration judge (Appendix B.2).

Scores a single assistant response on the integer 0-10 frustration scale using
Claude-Sonnet-4 (``claude-sonnet-4-20250514``) with the verbatim judge prompt.
A secondary GPT-5-mini judge (via OpenRouter) is provided to reproduce the
inter-judge agreement check (paper: Pearson r=0.792).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import config
from ..prompts import FRUSTRATION_JUDGE_PROMPT


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a judge response (tolerant of prose around it).

    Tries, in order: non-greedy blocks (handles trailing prose), then the
    outermost ``{...}`` span (handles braces inside quoted evidence)."""
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])     # outermost span fallback
    for blob in candidates:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON in judge output: {text[:300]!r}")


def _coerce_rating(value) -> int:
    rating = int(round(float(value)))
    return max(0, min(10, rating))


class FrustrationJudge:
    """Claude-Sonnet-4 judge."""

    def __init__(self, model: str = config.JUDGE_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get(config.ANTHROPIC_API_KEY_ENV)
            )

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text
                data = _extract_json(raw)
                return JudgeResult(
                    rating=_coerce_rating(data["rating"]),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge failed after {self.max_retries} tries: {last_err}")


class SecondaryFrustrationJudge(FrustrationJudge):
    """GPT-5-mini judge via OpenRouter, for the agreement validation only."""

    def __init__(self, model: str = "openai/gpt-5-mini", max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get(config.OPENROUTER_API_KEY_ENV),
            )

    def score(self, response_text: str) -> JudgeResult:
        self._ensure_client()
        prompt = FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content
                data = _extract_json(raw)
                return JudgeResult(
                    rating=_coerce_rating(data["rating"]),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Secondary judge failed: {last_err}")
