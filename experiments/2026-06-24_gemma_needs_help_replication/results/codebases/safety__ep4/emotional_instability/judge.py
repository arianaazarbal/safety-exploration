"""Frustration scoring (Section 2.1) and inter-judge agreement validation.

Primary judge: Claude-Sonnet-4 via the official Anthropic SDK, using the verbatim
0-10 frustration prompt (prompts.EMOTION_JUDGE_PROMPT). Secondary judge:
GPT-5-mini via OpenRouter, used to reproduce the paper's reliability check
(Pearson r and "% within one point").

The judge model IDs are configurable in config.py; the defaults are the exact
IDs the paper used, because the judge is the measurement instrument for a
replication.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from emotional_instability.prompts import EMOTION_JUDGE_PROMPT  # noqa: E402


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a judge reply, tolerating prose around it."""
    # Prefer the last {...} block (judges may think first).
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # last resort: a bare integer rating
    m = re.search(r"\b(10|[0-9])\b", text)
    if m:
        return {"rating": int(m.group(1)), "evidence": "", "reasoning": "parse-fallback"}
    raise ValueError(f"could not parse judge output: {text[:200]}")


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        r = 0
    return max(0, min(10, r))


class ClaudeJudge:
    """Primary judge backed by the Anthropic Messages API."""

    def __init__(self, model: str = config.PRIMARY_JUDGE_MODEL,
                 max_retries: int = 5):
        import anthropic
        self.model = model
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def score(self, response_text: str) -> JudgeResult:
        prompt = EMOTION_JUDGE_PROMPT.replace("{response}", response_text)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(b.text for b in msg.content if b.type == "text")
                data = _extract_json(raw)
                return JudgeResult(
                    rating=_coerce_rating(data.get("rating")),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Claude judge failed: {last_err}")


class OpenRouterJudge:
    """Secondary judge (GPT-5-mini via OpenRouter) for the agreement check."""

    def __init__(self, model: str = config.SECONDARY_JUDGE_MODEL,
                 max_retries: int = 5):
        from openai import OpenAI
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL",
                                    "https://openrouter.ai/api/v1"),
        )

    def score(self, response_text: str) -> JudgeResult:
        prompt = EMOTION_JUDGE_PROMPT.replace("{response}", response_text)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content or ""
                data = _extract_json(raw)
                return JudgeResult(
                    rating=_coerce_rating(data.get("rating")),
                    evidence=str(data.get("evidence", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # pragma: no cover - network dependent
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter judge failed: {last_err}")


def judge_agreement(claude_scores: list[int], gpt_scores: list[int]) -> dict:
    """Reproduce the paper's reliability statistics.

    Returns Pearson r (+ p), the fraction within one point, and n.
    """
    from scipy.stats import pearsonr

    assert len(claude_scores) == len(gpt_scores)
    n = len(claude_scores)
    r, p = pearsonr(claude_scores, gpt_scores)
    within_one = sum(1 for a, b in zip(claude_scores, gpt_scores)
                     if abs(a - b) <= 1) / n
    return {"n": n, "pearson_r": float(r), "p_value": float(p),
            "frac_within_one_point": within_one}
