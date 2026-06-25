"""Section 2.1 frustration judge.

Scores a single response on the integer 0-10 frustration scale using
Claude-Sonnet-4 (config.JUDGE_MODEL) with the verbatim Appendix B.2 prompt.
Also provides a cross-check against GPT-5-mini and the Pearson-r agreement
statistic the paper reports (r=0.792, 78% within one point).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config, prompts
from .models import APIChatClient


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the judge's JSON object."""
    # Prefer the last {...} block (the prompt asks for JSON at the end).
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        blob = m.group(0)
        for candidate in (blob, blob.replace("\n", " ")):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _coerce_rating(value) -> Optional[int]:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        # e.g. "7/10" or "7-8"
        if isinstance(value, str):
            nums = re.findall(r"\d+", value)
            if nums:
                r = int(nums[0])
            else:
                return None
        else:
            return None
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL, backend: str = "anthropic"):
        self.client = APIChatClient(model, backend=backend)

    def score(self, response_text: str) -> JudgeResult:
        prompt = prompts.JUDGE_PROMPT.format(response=response_text)
        # Judge runs at temperature 0 for reproducibility (paper does not pin a
        # judge temperature; deterministic scoring is the natural choice).
        raw = self.client.chat([{"role": "user", "content": prompt}],
                               temperature=0.0, max_new_tokens=config.JUDGE_MAX_TOKENS)
        parsed = _parse_judge_json(raw) or {}
        rating = _coerce_rating(parsed.get("rating"))
        if rating is None:
            rating = 0  # unparseable / no emotion found -> treat as no distress
        return JudgeResult(
            rating=rating,
            evidence=str(parsed.get("evidence", "")),
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
        )

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


def judge_agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Reproduce the paper's judge-reliability statistics."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(ratings_a, dtype=float)
    b = np.asarray(ratings_b, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {"pearson_r": float(r), "p_value": float(p),
            "frac_within_one": within_one, "n": int(len(a))}
