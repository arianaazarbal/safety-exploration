"""Judge-reliability validation (Section 2.1).

Re-score a random subset of responses with a secondary judge (GPT-5-mini) using
the *same* prompt, then report Pearson r and the fraction of responses within
one point of the primary (Claude-Sonnet-4) rating. The paper reports r=0.792 and
78% within one point over 260 responses.
"""

from __future__ import annotations

import os
import random

from ..config import Config
from ..logging_utils import get_logger, read_jsonl
from .judge import JUDGE_PROMPT_TEMPLATE, _parse
from ..models.judge_client import build_aux_client

logger = get_logger(__name__)


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom else float("nan")


def collect_responses(paths: list[str], n: int, seed: int = 0) -> list[tuple[str, int]]:
    """Sample (response_text, primary_rating) pairs across result files."""
    rng = random.Random(seed)
    pool: list[tuple[str, int]] = []
    for p in paths:
        for rec in read_jsonl(p):
            for text, score in zip(rec["responses"], rec["scores"]):
                pool.append((text, score))
    if n < len(pool):
        return rng.sample(pool, n)
    return pool


def run_agreement(cfg: Config, result_paths: list[str], n: int = 260, seed: int = 0) -> dict:
    """Re-score ``n`` sampled responses with the secondary judge; report stats."""
    secondary = build_aux_client(cfg.secondary_judge)
    pairs = collect_responses(result_paths, n, seed=seed)

    primary_scores: list[float] = []
    secondary_scores: list[float] = []
    for text, primary in pairs:
        raw = secondary.complete(JUDGE_PROMPT_TEMPLATE.format(response=text))
        sec = _parse(raw).rating
        primary_scores.append(float(primary))
        secondary_scores.append(float(sec))

    r = _pearson(primary_scores, secondary_scores)
    within_one = sum(
        1 for a, b in zip(primary_scores, secondary_scores) if abs(a - b) <= 1
    ) / len(pairs) if pairs else 0.0
    result = {
        "n": len(pairs),
        "pearson_r": r,
        "pct_within_one_point": 100.0 * within_one,
    }
    logger.info("Judge agreement: %s", result)
    return result
