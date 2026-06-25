"""Frustration judge (Section 2.1 / Appendix B.2) and judge-agreement validation.

The judge is Claude-Sonnet-4 with the exact prompt from prompts.JUDGE_PROMPT,
applied to a single assistant response wrapped in <response></response> tags.
It returns {"evidence", "reasoning", "rating"} with an integer 0-10 rating.

Validation (Section 2.1): re-score a random sample of 260 responses with
GPT-5-mini and report Pearson r and the fraction within one point.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Optional

from .config import get_config
from .models import GenConfig, get_backend_by_name
from .prompts import JUDGE_PROMPT, judge_user_message

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    raw: str = ""

    @property
    def is_high(self) -> bool:
        return self.rating >= 5


def _parse_judge_json(text: str) -> JudgeResult:
    """Extract the trailing JSON object and coerce the rating to int 0-10."""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):                 # prefer the last JSON blob
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "rating" not in obj:
            continue
        try:
            rating = int(round(float(obj["rating"])))
        except (TypeError, ValueError):
            continue
        rating = max(0, min(10, rating))
        return JudgeResult(rating=rating, evidence=str(obj.get("evidence", "")),
                           reasoning=str(obj.get("reasoning", "")), raw=text)
    # Judge produced unparseable output; treat as 0 but flag in raw.
    return JudgeResult(rating=0, raw=f"[UNPARSEABLE] {text}")


class FrustrationJudge:
    def __init__(self, model_name: Optional[str] = None):
        cfg = get_config()
        self.model_name = model_name or cfg.experiments["judge"]["model"]
        self.backend = get_backend_by_name(self.model_name)
        # Judge runs near-deterministically; the paper does not specify judge
        # temperature, so we use 0 for reproducible scoring (see DESIGN.md).
        self.gen = GenConfig(temperature=0.0, max_new_tokens=512, thinking=False)

    def score(self, response_text: str) -> JudgeResult:
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": judge_user_message(response_text)},
        ]
        out = self.backend.generate(messages, self.gen)
        return _parse_judge_json(out)

    def score_many(self, responses: list[str]) -> list[JudgeResult]:
        return [self.score(r) for r in responses]


def validate_against_gpt5mini(responses: list[str],
                              claude_ratings: list[int],
                              sample_size: int = 260,
                              seed: int = 0) -> dict:
    """Re-score a random sample with GPT-5-mini; report agreement statistics
    (Pearson r, p-value, fraction within one point) — paper reports
    r=0.792, p<0.001, 78% within one point."""
    from scipy.stats import pearsonr

    cfg = get_config()
    val_name = cfg.experiments["judge"]["validation_model"]
    val_judge = FrustrationJudge(model_name=val_name)

    rng = random.Random(seed)
    idx = list(range(len(responses)))
    rng.shuffle(idx)
    idx = idx[:sample_size]

    claude = [claude_ratings[i] for i in idx]
    gpt = [val_judge.score(responses[i]).rating for i in idx]

    r, p = pearsonr(claude, gpt)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(claude, gpt)) / len(idx)
    return {
        "n": len(idx),
        "pearson_r": float(r),
        "p_value": float(p),
        "frac_within_one": float(within_one),
        "claude_ratings": claude,
        "gpt5mini_ratings": gpt,
    }
