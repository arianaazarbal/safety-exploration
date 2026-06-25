"""Judge-reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports
Pearson r = 0.792 (p < 0.001) and 78% of responses within one point of the
Claude-Sonnet ratings. This module samples 260 already-judged responses, scores
them with the cross-check judge, and computes those two statistics.
"""

from __future__ import annotations

import random
import statistics

from .. import config
from ..eval.judge import CrossCheckJudge
from ..eval.run_eval import responses_path
from ..utils.io import read_jsonl


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def run_crosscheck(model_keys: list[str], n: int = config.CROSSCHECK_SAMPLE_SIZE,
                   seed: int = 0) -> dict:
    rows = []
    for key in model_keys:
        rows.extend(read_jsonl(responses_path(key)))
    rng = random.Random(seed)
    rng.shuffle(rows)
    sample = rows[:n]

    cross = CrossCheckJudge()
    primary_scores, cross_scores = [], []
    for r in sample:
        primary_scores.append(float(r["rating"]))
        cross_scores.append(float(cross.score(r["response"]).rating))

    r = _pearson(primary_scores, cross_scores)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary_scores, cross_scores)) / len(sample)
    return {"n": len(sample), "pearson_r": r, "within_one_point": within_one}
