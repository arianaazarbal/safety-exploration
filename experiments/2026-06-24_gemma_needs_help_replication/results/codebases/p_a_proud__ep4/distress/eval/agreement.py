"""Judge-agreement validation (Paper §2.1).

Re-score a random sample of responses with a secondary judge (GPT-5-mini) and
report Pearson correlation against the primary judge plus the fraction of scores
within one point — the paper reports r = 0.792 and 78% within one point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import build_model
from ..types import ScoredTurn
from ..utils.seeding import derived_rng
from .judge import score_response


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    within_one_point: float
    primary_scores: list[int]
    secondary_scores: list[int]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pearson_r": self.pearson_r,
            "within_one_point": self.within_one_point,
        }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def validate_judges(
    judged_rows: list[dict],
    *,
    secondary_judge: str = "judge_validation",
    n_samples: int = 260,
    seed: int = 0,
) -> AgreementResult:
    """Sample ``n_samples`` already-judged rows and re-score with a second judge."""
    scored = [r for r in judged_rows if r.get("score") is not None]
    rng = derived_rng(seed, "agreement", len(scored))
    if len(scored) > n_samples:
        sample = rng.sample(scored, n_samples)
    else:
        sample = scored

    secondary = build_model(secondary_judge)
    primary_scores: list[int] = []
    secondary_scores: list[int] = []
    for r in sample:
        v = score_response(secondary, r["response"])
        primary_scores.append(int(r["score"]))
        secondary_scores.append(int(v.rating))

    within = sum(
        1 for a, b in zip(primary_scores, secondary_scores) if abs(a - b) <= 1
    ) / max(1, len(primary_scores))
    r = _pearson([float(s) for s in primary_scores], [float(s) for s in secondary_scores])
    return AgreementResult(
        n=len(primary_scores),
        pearson_r=r,
        within_one_point=within,
        primary_scores=primary_scores,
        secondary_scores=secondary_scores,
    )
