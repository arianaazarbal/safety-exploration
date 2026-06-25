"""Validate judge reliability by re-scoring a random sample with a second judge.

paper Section 2.1: "we randomly sampled 260 responses for re-scoring with GPT-5-mini,
using the same prompt. The judges show strong agreement (Pearson r = 0.792, p < 0.001),
with 78% of responses within one point."

This script samples N already-scored responses across all models, re-scores them with the
second judge (GPT-5-mini via OpenRouter), and reports Pearson r and the within-1-point
rate, mirroring the paper's check.
"""

from __future__ import annotations

import asyncio
import json
import os
import random

import config
from analyze import load_records
from clients import OpenRouterJudgeClient


async def validate(n: int = 260, seed: int = config.SEED) -> dict:
    pool: list[dict] = []
    for m in config.TARGET_MODELS:
        pool.extend(load_records(m.key))
    if not pool:
        raise RuntimeError("No scored responses found; run the evaluation first.")

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    judge2 = OpenRouterJudgeClient()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_ROLLOUTS)

    async def rescore(rec):
        async with sem:
            try:
                v = await judge2.score(rec["response"])
                return rec["rating"], v["rating"]
            except Exception as exc:  # noqa: BLE001
                print(f"re-score failed: {exc}")
                return None

    pairs = [p for p in await asyncio.gather(*[rescore(r) for r in sample]) if p]
    if len(pairs) < 2:
        raise RuntimeError("Too few successful re-scores to compute agreement.")

    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]

    from scipy.stats import pearsonr
    r, p = pearsonr(a, b)
    within1 = 100.0 * sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)

    result = {"n": len(pairs), "pearson_r": r, "p_value": p, "within_1_point_pct": within1,
              "paper_pearson_r": 0.792, "paper_within_1_pct": 78.0}

    os.makedirs(config.TABLES_DIR, exist_ok=True)
    with open(os.path.join(config.TABLES_DIR, "judge_agreement.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== Judge agreement (vs GPT-5-mini) ===")
    print(f"n={result['n']}  Pearson r={r:.3f} (paper 0.792)  "
          f"p={p:.2e}  within-1-point={within1:.1f}% (paper 78%)")
    return result
