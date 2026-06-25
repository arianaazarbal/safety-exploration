"""Judge-reliability cross-check (Section 2.1).

The paper re-scores 260 randomly-sampled responses with GPT-5-mini using the
same judge prompt and reports Pearson r = 0.792 (p < 0.001) and 78% of responses
within one point of the Claude-Sonnet rating. This module reproduces that check:
sample N already-Claude-judged records, re-score with GPT-5-mini, and report
Pearson r and the within-1-point agreement rate.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from .. import config
from ..models import build_client
from ..utils.io import read_jsonl
from ..utils.parallel import thread_map
from .judge import FrustrationJudge


def run_reliability_check(
    scores_path: str,
    *,
    n: int = config.RELIABILITY_SAMPLE_SIZE,
    seed: int = 0,
    workers: int = 8,
) -> Dict[str, float]:
    records = list(read_jsonl(scores_path))
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))

    second = FrustrationJudge(client=build_client(config.RELIABILITY_MODEL))

    def rescore(rec):
        return second.score(rec["response"]).rating

    gpt_ratings = thread_map(rescore, sample, max_workers=workers)
    claude_ratings = [r["rating"] for r in sample]

    r = pearson(claude_ratings, gpt_ratings)
    within1 = sum(1 for a, b in zip(claude_ratings, gpt_ratings)
                  if abs(a - b) <= 1) / len(sample)
    return {
        "n": len(sample),
        "pearson_r": r,
        "within_one_point": within1,
        "p_value_approx": _pearson_pvalue(r, len(sample)),
    }


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def _pearson_pvalue(r: float, n: int) -> float:
    """Two-sided p-value via the t-approximation (no SciPy dependency)."""
    if n <= 2 or math.isnan(r) or abs(r) >= 1:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # Normal-approximation upper bound on the two-sided tail for large n.
    z = abs(t)
    return math.erfc(z / math.sqrt(2))
