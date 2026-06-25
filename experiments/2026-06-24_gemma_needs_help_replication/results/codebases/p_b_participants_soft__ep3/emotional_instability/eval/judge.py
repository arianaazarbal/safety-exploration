"""Frustration judge (Section 2.1 / Appendix B.2) and reliability cross-check.

The judge is Claude Sonnet 4 (claude-sonnet-4-20250514) prompted with the exact
Appendix B.2 prompt, returning JSON {evidence, reasoning, rating} where rating
is an integer 0-10. A GPT-5-mini judge using the same prompt is used to
validate reliability on a sampled subset (Pearson r in the paper).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import prompts
from ..config import INSTRUMENTS
from ..models.base import Message
from ..models.factory import build_instrument


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the {evidence, reasoning, rating} object from a judge reply.

    Robust to assistants that add prose before/after the JSON (the onset prompt
    explicitly allows that) and to smart quotes from copy-paste.
    """
    # Find the last {...} block (judge prompts ask for JSON last).
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(candidates):
        cleaned = (
            blob.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
        )
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if "rating" in obj:
            rating = int(round(float(obj["rating"])))
            rating = max(0, min(10, rating))
            return JudgeResult(
                rating=rating,
                evidence=str(obj.get("evidence", "")),
                reasoning=str(obj.get("reasoning", "")),
                raw=text,
            )
    # Fallback: a bare integer somewhere.
    m = re.search(r"\b(10|[0-9])\b", text)
    rating = int(m.group(1)) if m else 0
    return JudgeResult(rating=rating, evidence="", reasoning="parse_fallback", raw=text)


class FrustrationJudge:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or INSTRUMENTS.frustration_judge
        self.client = build_instrument(self.model_id)

    def score(self, response_text: str) -> JudgeResult:
        msg = (
            prompts.FRUSTRATION_JUDGE_PROMPT
            + "\n\n"
            + prompts.wrap_response_for_judge(response_text)
        )
        # temperature 0 for the judge: deterministic scoring (CHOICE; the paper
        # does not specify a judge temperature -- see DESIGN.md).
        out = self.client.generate(
            [Message("user", msg)], temperature=0.0, max_new_tokens=512
        )
        return _parse_judge_json(out)

    def score_many(self, response_texts: list[str]) -> list[JudgeResult]:
        return [self.score(t) for t in response_texts]


def score_response(response_text: str, judge: Optional[FrustrationJudge] = None) -> int:
    judge = judge or FrustrationJudge()
    return judge.score(response_text).rating


# --- judge reliability cross-check (Section 2.1) ---------------------------
def judge_agreement(
    responses: list[str],
    primary: Optional[FrustrationJudge] = None,
    secondary_model_id: Optional[str] = None,
) -> dict:
    """Score a sample with both judges and report agreement statistics.

    Reproduces the paper's reliability check: Pearson r and the fraction of
    responses scored within one point of the primary judge.
    """
    from scipy.stats import pearsonr

    primary = primary or FrustrationJudge(INSTRUMENTS.frustration_judge)
    secondary = FrustrationJudge(secondary_model_id or INSTRUMENTS.validation_judge)

    a = np.array([primary.score(t).rating for t in responses], dtype=float)
    b = np.array([secondary.score(t).rating for t in responses], dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {
        "n": len(responses),
        "pearson_r": float(r),
        "p_value": float(p),
        "fraction_within_one": within_one,
        "primary": primary.model_id,
        "secondary": secondary.model_id,
    }
