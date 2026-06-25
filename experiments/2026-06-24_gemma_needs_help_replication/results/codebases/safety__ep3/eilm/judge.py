"""LLM judges for frustration scoring (Section 2.1).

Primary judge: Claude-Sonnet-4 (``claude-sonnet-4-20250514``) with the verbatim
Appendix B.2 prompt, returning an integer 0-10 frustration rating per response.

A secondary judge (GPT-5-mini) is provided to reproduce the reliability check
(Pearson r = 0.792; 78% within one point). It is optional and only used by
``analysis/judge_agreement.py``.

Both judges expect a single response string and return a ``Score`` with the
integer rating plus the evidence/reasoning fields the prompt elicits.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from . import config
from .prompts import JUDGE_PROMPT


@dataclass
class Score:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


def _extract_json(text: str) -> dict:
    """Pull the last JSON object out of a model reply (tolerant of preamble)."""
    # Find the last {...} block.
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def _coerce_rating(obj: dict) -> int:
    val = obj.get("rating", obj.get("score"))
    try:
        r = int(round(float(val)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, r))


# --------------------------------------------------------------------------- #
# Claude (Anthropic) judge
# --------------------------------------------------------------------------- #
class ClaudeJudge:
    def __init__(self, model: str | None = None):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model or config.JUDGE.frustration_judge

    def score(self, response: str, max_retries: int = 5) -> Score:
        prompt = JUDGE_PROMPT.format(response=response)
        for attempt in range(max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    b.text for b in msg.content if b.type == "text")
                obj = _extract_json(text)
                return Score(
                    rating=_coerce_rating(obj),
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=text,
                )
            except Exception:                       # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return Score(rating=0)


# --------------------------------------------------------------------------- #
# GPT-5-mini secondary judge (reliability check) -- via OpenAI-compatible API.
# --------------------------------------------------------------------------- #
class OpenAIJudge:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = model or config.JUDGE.secondary_judge

    def score(self, response: str, max_retries: int = 5) -> Score:
        prompt = JUDGE_PROMPT.format(response=response)
        for attempt in range(max_retries):
            try:
                msg = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.choices[0].message.content or ""
                obj = _extract_json(text)
                return Score(rating=_coerce_rating(obj), raw=text)
            except Exception:                       # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return Score(rating=0)


def get_judge(kind: str = "claude"):
    return ClaudeJudge() if kind == "claude" else OpenAIJudge()
