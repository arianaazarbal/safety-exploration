"""Inter-rater reliability check (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports
Pearson r = 0.792 (p < 0.001) and 78% of responses within one point of the
Claude-Sonnet ratings. This module re-scores a random subset with the validation
judge and computes the same statistics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..models import GenConfig, get_client
from ..prompts.judge_prompts import build_judge_prompt
from .parsing import parse_verdict


@dataclass
class ReliabilityResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float
    primary: list[int]
    secondary: list[int]


def validate_judges(
    scored_turns: list[dict],
    *,
    n_resample: int = 260,
    validation_model: str = "validation_judge",
    seed: int = 0,
) -> ReliabilityResult:
    """``scored_turns`` are dict records from :func:`frustration_judge.read_scores`
    (must contain ``text`` and primary ``rating``)."""
    from scipy.stats import pearsonr

    usable = [t for t in scored_turns if t.get("rating") is not None]
    rng = random.Random(seed)
    sample = rng.sample(usable, min(n_resample, len(usable)))

    judge = get_client(validation_model)
    cfg = GenConfig(temperature=0.0, max_tokens=512, top_p=1.0)

    primary: list[int] = []
    secondary: list[int] = []
    for t in sample:
        out = judge.generate(
            [{"role": "user", "content": build_judge_prompt(t["text"])}], cfg
        )
        v = parse_verdict(out)
        if v.rating is None:
            continue
        primary.append(int(t["rating"]))
        secondary.append(int(v.rating))

    r, p = pearsonr(primary, secondary)
    within = sum(1 for a, b in zip(primary, secondary) if abs(a - b) <= 1)
    pct = within / len(primary) if primary else float("nan")
    return ReliabilityResult(
        n=len(primary), pearson_r=float(r), p_value=float(p),
        pct_within_one=pct, primary=primary, secondary=secondary,
    )
