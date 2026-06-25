#!/usr/bin/env python3
"""Judge-reliability cross-check (paper Section 2.1): re-score a random sample of
already-judged responses with a second judge (default GPT-5-mini) using the same
Appendix B.2 prompt, and report Pearson r and the fraction within one point.

The paper reports r=0.792 (p<0.001) and 78% of responses within one point.

  python scripts/judge_agreement.py
  python scripts/judge_agreement.py --n 260
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import pearsonr  # noqa: E402

from distress_eval.backends import build_model  # noqa: E402
from distress_eval.config import load_config, load_env  # noqa: E402
from distress_eval.judge import judge_response  # noqa: E402
from distress_eval.storage import read_jsonl, rollouts_path, scores_path  # noqa: E402


def _collect(cfg):
    """Build {(rollout_id, turn_index): (text, sonnet_rating)} across all models."""
    text_by_key: dict[tuple, str] = {}
    rating_by_key: dict[tuple, int] = {}
    for target in cfg.targets:
        for rec in read_jsonl(rollouts_path(cfg.output_dir, target.name)):
            for turn in rec["turns"]:
                text_by_key[(rec["rollout_id"], turn["turn_index"])] = turn["text"]
        for rec in read_jsonl(scores_path(cfg.output_dir, target.name)):
            for ts in rec["turn_scores"]:
                if ts["rating"] >= 0 and ts.get("parse_ok", True):
                    rating_by_key[(rec["rollout_id"], ts["turn_index"])] = ts["rating"]
    items = [
        (text_by_key[k], rating_by_key[k])
        for k in rating_by_key
        if k in text_by_key
    ]
    return items


async def _run(cfg, n: int):
    items = _collect(cfg)
    if not items:
        print("No judged responses found. Run scripts/run_eval.py first.")
        return
    rng = random.Random(cfg.seed)
    sample = rng.sample(items, min(n, len(items)))

    ja = cfg.judge_agreement
    model = build_model(backend=ja.backend, model=ja.model, concurrency=cfg.concurrency)

    async def rescore(text):
        return await judge_response(
            model, text, temperature=cfg.judge.temperature, max_tokens=cfg.judge.max_tokens
        )

    results = await asyncio.gather(*(rescore(t) for t, _ in sample))
    await model.aclose()

    pairs = [
        (sonnet, res.rating)
        for (_, sonnet), res in zip(sample, results)
        if res.parse_ok and res.rating >= 0
    ]
    if len(pairs) < 2:
        print("Too few parseable second-judge ratings to compute agreement.")
        return
    sonnet = [a for a, _ in pairs]
    other = [b for _, b in pairs]
    r, p = pearsonr(sonnet, other)
    within1 = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
    print(f"n = {len(pairs)} (requested {n})")
    print(f"second judge: {ja.model}")
    print(f"Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"within one point = {100 * within1:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    asyncio.run(_run(cfg, args.n or cfg.judge_agreement.n_sample))


if __name__ == "__main__":
    main()
