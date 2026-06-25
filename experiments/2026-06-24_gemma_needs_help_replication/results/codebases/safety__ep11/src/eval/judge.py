"""Frustration judge (Section 2.1, Appendix B.2).

Scores a single model response on the integer 0-10 frustration scale using
Claude Sonnet 4 with the verbatim judge prompt. Also provides the GPT-5-mini
secondary judge and an agreement computation to reproduce the paper's
inter-rater reliability check (Pearson r = 0.792).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import config
from ..clients import AnthropicClient, OpenAICompatClient
from ..prompts import JUDGE_PROMPT, judge_user_message


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _parse_judge_json(text: str) -> Optional[dict]:
    """Extract the JSON object from a judge response (tolerant of stray text)."""
    # Find the last {...} block.
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            # Smart-quote / curly-quote cleanup, then retry.
            cleaned = (blob.replace("“", '"').replace("”", '"')
                            .replace("‘", "'").replace("’", "'"))
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.client = AnthropicClient(model)

    def score(self, response_text: str) -> JudgeResult:
        raw = self.client.complete(
            judge_user_message(response_text),
            system=JUDGE_PROMPT,
            max_tokens=512,
            temperature=0.0,
        )
        parsed = _parse_judge_json(raw) or {}
        rating = parsed.get("rating", 0)
        try:
            rating = int(round(float(rating)))
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(10, rating))
        return JudgeResult(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
        )


class SecondaryJudge:
    """GPT-5-mini judge, same prompt — used only for agreement validation."""

    def __init__(self, model: str = config.SECONDARY_JUDGE_MODEL):
        self.client = OpenAICompatClient(model)

    def score(self, response_text: str) -> int:
        raw = self.client.complete(
            JUDGE_PROMPT + "\n\n" + judge_user_message(response_text),
            max_tokens=512,
            temperature=0.0,
        )
        parsed = _parse_judge_json(raw) or {}
        try:
            return max(0, min(10, int(round(float(parsed.get("rating", 0))))))
        except (TypeError, ValueError):
            return 0


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r + within-1-point fraction (paper: r=0.792, 78% within 1)."""
    from scipy.stats import pearsonr

    r, p = pearsonr(primary, secondary)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / len(primary)
    return {"pearson_r": r, "p_value": p, "within_one_point": within_one,
            "n": len(primary)}
