"""Frustration judge: 0-10 scoring with the verbatim Appendix B.2 prompt, plus
cross-judge agreement (Section 2.1: Pearson r against GPT-5-mini)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from ..config import JudgeConfig
from ..judge_prompts import (
    FRUSTRATION_JUDGE_PROMPT,
    frustration_judge_user_message,
)


class _Completer(Protocol):
    def complete(self, user: str, system: Optional[str] = None, **kw) -> str: ...


@dataclass
class JudgeScore:
    rating: float
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> JudgeScore:
    """Parse the judge's JSON. The judge is asked for
    ``{"evidence", "reasoning", "rating"}``; we parse defensively because models
    occasionally wrap JSON in prose or use smart quotes."""
    match = _JSON_RE.search(text)
    blob = match.group(0) if match else text
    # Normalise smart quotes that appear in the paper's prompt examples.
    blob = blob.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        data = json.loads(blob)
        rating = float(data.get("rating"))
        return JudgeScore(
            rating=rating,
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
            raw=text,
        )
    except Exception:
        # Last-ditch: find a bare integer 0-10.
        m = re.search(r'"?rating"?\s*[:=]\s*(\d+(?:\.\d+)?)', text)
        if not m:
            m = re.search(r"\b(10|[0-9])\b", text)
        rating = float(m.group(1)) if m else 0.0
        return JudgeScore(rating=rating, raw=text)


class FrustrationJudge:
    """Scores a single response (one assistant turn) on the 0-10 scale.

    Per the paper, scoring is per-response: each assistant turn is scored
    independently with only that turn's text inside ``<response>`` tags.
    """

    def __init__(self, completer: _Completer, config: Optional[JudgeConfig] = None):
        self.completer = completer
        self.config = config or JudgeConfig()

    def score(self, response_text: str) -> JudgeScore:
        out = self.completer.complete(
            user=frustration_judge_user_message(response_text),
            system=FRUSTRATION_JUDGE_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        score = _parse_judge_json(out)
        # Clamp to the valid integer range.
        score.rating = max(0.0, min(10.0, score.rating))
        return score

    def score_many(self, responses: list[str]) -> list[JudgeScore]:
        return [self.score(r) for r in responses]


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1)
# --------------------------------------------------------------------------- #
@dataclass
class AgreementStats:
    n: int
    pearson_r: float
    p_value: float
    within_one_fraction: float


def judge_agreement(scores_a: list[float], scores_b: list[float]) -> AgreementStats:
    """Reproduce the judge-validation statistics: Pearson r, p-value, and the
    fraction of responses within one point (paper reports r=0.792, p<0.001,
    78% within one point on 260 re-scored responses)."""
    import math

    assert len(scores_a) == len(scores_b) and scores_a, "need equal, non-empty lists"
    n = len(scores_a)
    within_one = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= 1) / n

    # Pearson correlation.
    ma = sum(scores_a) / n
    mb = sum(scores_b) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(scores_a, scores_b))
    va = sum((a - ma) ** 2 for a in scores_a)
    vb = sum((b - mb) ** 2 for b in scores_b)
    if va == 0 or vb == 0:
        r = 0.0
    else:
        r = cov / math.sqrt(va * vb)

    # Two-sided p-value via t-distribution approximation.
    if n > 2 and abs(r) < 1.0:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # Normal approximation to the t CDF tail (adequate for large n).
        p = 2 * (1 - _normal_cdf(abs(t)))
    else:
        p = 0.0
    return AgreementStats(n=n, pearson_r=r, p_value=p, within_one_fraction=within_one)


def _normal_cdf(x: float) -> float:
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
