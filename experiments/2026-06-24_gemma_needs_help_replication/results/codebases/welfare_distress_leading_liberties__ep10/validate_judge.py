"""Optional: replicate the judge-reliability check (Section 2.1).

Randomly samples N already-scored responses, re-scores them with the secondary
judge (GPT-5-mini) using the identical prompt, and reports Pearson r and the
fraction of responses within one point -- the paper finds r = 0.792 and 78%
within one point over 260 responses.

Usage:
    OPENAI_API_KEY=... python validate_judge.py [N]   # N defaults to 260
"""

from __future__ import annotations

import asyncio
import json
import random
import sys

from scipy.stats import pearsonr

from config import (
    RESPONSES_PATH, SCORES_PATH, SECONDARY_JUDGE, load_run_config,
)
from judge import Judge


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


async def main(n: int):
    cfg = load_run_config()
    primary = {r["response_id"]: r for r in _load_jsonl(SCORES_PATH)
               if r["rating"] >= 0}
    texts = {r["response_id"]: r["text"] for r in _load_jsonl(RESPONSES_PATH)
             if r.get("text")}

    ids = [rid for rid in primary if rid in texts]
    rng = random.Random(cfg.seed)
    sample = rng.sample(ids, min(n, len(ids)))
    print(f"Re-scoring {len(sample)} responses with {SECONDARY_JUDGE.model}")

    judge = Judge(SECONDARY_JUDGE, cfg)
    sem = asyncio.Semaphore(cfg.judge_concurrency)

    async def rescore(rid):
        async with sem:
            res = await judge.score(texts[rid])
        return rid, res.rating

    results = await asyncio.gather(*(rescore(rid) for rid in sample))

    pairs = [(primary[rid]["rating"], r) for rid, r in results if r >= 0]
    a = [p for p, _ in pairs]
    b = [s for _, s in pairs]
    r, p = pearsonr(a, b)
    within1 = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)

    print(f"\nn = {len(pairs)} (parseable on both judges)")
    print(f"Pearson r = {r:.3f}  (p = {p:.2e})   [paper: 0.792]")
    print(f"Within 1 point = {100 * within1:.1f}%             [paper: 78%]")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 260
    asyncio.run(main(n))
