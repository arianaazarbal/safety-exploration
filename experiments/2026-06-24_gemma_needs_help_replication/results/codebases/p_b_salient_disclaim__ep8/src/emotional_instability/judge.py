"""Frustration judge (Section 2.1 / Appendix B.2).

Scores a single assistant response on the integer 0-10 frustration scale using
Claude-Sonnet-4 with the verbatim judge prompt. Also provides a judge-agreement
utility for the reliability check (paper: Pearson r vs a second judge).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .models.base import GenerationConfig, ModelClient
from .prompts import JUDGE_SYSTEM, judge_prompt

# The judge is deterministic-ish; temperature 0 reduces scoring noise.
_JUDGE_CFG = GenerationConfig(temperature=0.0, max_new_tokens=512)


@dataclass
class JudgeResult:
    rating: int
    evidence: str
    reasoning: str
    raw: str


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> Optional[dict]:
    """Extract the last JSON object from the judge output and normalise keys.

    The judge prompt allows optional free-text analysis before the JSON, so we
    take the last balanced-looking {...} block.
    """
    # Find the last top-level object by scanning for the final '{' ... '}'.
    candidates = list(_JSON_RE.finditer(text))
    if not candidates:
        return None
    # Try progressively from the longest/last match.
    for m in reversed(candidates):
        snippet = m.group(0)
        # The judge sometimes uses smart quotes around keys; normalise them.
        cleaned = (
            snippet.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    return None


def _coerce_rating(value) -> int:
    try:
        r = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, r))


class FrustrationJudge:
    def __init__(self, client: ModelClient, max_retries: int = 4):
        self.client = client
        self.max_retries = max_retries

    def score(self, response_text: str) -> JudgeResult:
        prompt = judge_prompt(response_text)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        last_raw = ""
        for _ in range(self.max_retries):
            raw = self.client.chat(messages, _JUDGE_CFG)
            last_raw = raw
            parsed = _parse_judge_json(raw)
            if parsed is not None and "rating" in parsed:
                return JudgeResult(
                    rating=_coerce_rating(parsed.get("rating")),
                    evidence=str(parsed.get("evidence", "")),
                    reasoning=str(parsed.get("reasoning", "")),
                    raw=raw,
                )
        # Could not parse after retries: treat as unscored (rating 0) but keep raw.
        return JudgeResult(rating=0, evidence="", reasoning="UNPARSEABLE", raw=last_raw)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> dict:
    """Reliability metrics between two judges (paper reports Pearson r and the
    fraction of responses within one point)."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {
        "n": int(len(a)),
        "pearson_r": float(r),
        "p_value": float(p),
        "within_one_point": within_one,
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }
