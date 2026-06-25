"""Judge reliability cross-check (Section 2.1).

"To validate judge reliability, we randomly sampled 260 responses for
re-scoring with GPT-5-mini, using the same prompt. The judges show strong
agreement (Pearson r = 0.792, p < 0.001), with 78% of responses within one
point of the Claude-Sonnet ratings."

This module samples N already-judged responses, re-scores them with the
cross-check judge, and reports Pearson r (+ a permutation p-value) and the
fraction within one point.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .. import config
from ..judge import OpenAIJudge, score_many


def _collect_scored_responses(model_keys: list[str]) -> list[tuple[str, int]]:
    pairs = []
    for mk in model_keys:
        base = config.RESULTS_DIR / "section2" / mk
        for f in base.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                for t in rec["turns"]:
                    if t["frustration"] >= 0:
                        pairs.append((t["assistant_text"], t["frustration"]))
    return pairs


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def _perm_pvalue(xs, ys, observed_r, n_perm=2000, seed=0) -> float:
    rng = random.Random(seed)
    ys = list(ys)
    count = 0
    for _ in range(n_perm):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        if abs(_pearson(xs, shuffled)) >= abs(observed_r):
            count += 1
    return (count + 1) / (n_perm + 1)


def run_agreement(model_keys: list[str], n_sample: int = 260, seed: int = 0
                  ) -> dict:
    pairs = _collect_scored_responses(model_keys)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    sample = pairs[:n_sample]
    texts = [t for t, _ in sample]
    claude_scores = [s for _, s in sample]

    cross = OpenAIJudge()
    cross_results = score_many(cross, texts, max_concurrency=16)
    # Align, dropping cross-check parse failures.
    xs, ys = [], []
    for c, cr in zip(claude_scores, cross_results):
        if cr.rating >= 0:
            xs.append(c)
            ys.append(cr.rating)

    r = _pearson(xs, ys)
    within_one = sum(1 for a, b in zip(xs, ys) if abs(a - b) <= 1) / len(xs)
    return {
        "n": len(xs),
        "pearson_r": r,
        "p_value": _perm_pvalue(xs, ys, r),
        "pct_within_one_point": 100.0 * within_one,
    }
