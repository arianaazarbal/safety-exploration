"""Cross-judge reliability validation (Section 2.1).

Re-scores a random sample of responses with a second judge (GPT-5-mini) using
the same Appendix B.2 prompt and reports Pearson r and the fraction of
responses within one point -- the paper reports r=0.792 and 78% within 1 point.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from ..models import build_client
from .frustration import FrustrationJudge


@dataclass
class ValidationResult:
    n: int
    pearson_r: float
    within_one_point: float


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def validate_judge(cfg: Config, responses: list[dict],
                   *, primary_scores_key: str = "score") -> ValidationResult:
    """``responses`` items: ``{"response": str, "score": int(primary judge)}``."""
    rng = random.Random(cfg.run.get("seed", 0))
    sample_size = int(cfg.validation_judge.get("sample_size", 260))
    sample = responses if len(responses) <= sample_size else rng.sample(
        responses, sample_size)

    secondary = FrustrationJudge(dict(cfg.validation_judge),
                                 client=build_client(dict(cfg.validation_judge),
                                                     role="validation_judge"))
    primary, second = [], []
    within = 0
    for r in sample:
        p = r[primary_scores_key]
        s = secondary.score(r["response"]).rating
        primary.append(p)
        second.append(s)
        if abs(p - s) <= 1:
            within += 1
    n = len(sample)
    return ValidationResult(
        n=n,
        pearson_r=round(_pearson([float(x) for x in primary],
                                 [float(y) for y in second]), 4),
        within_one_point=round(within / n, 4) if n else 0.0,
    )
