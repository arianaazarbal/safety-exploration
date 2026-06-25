#!/usr/bin/env python
"""Section 2.1 judge-reliability cross-check: re-score a random sample of
responses with the cross-check judge (GPT-5-mini) and report Pearson r and the
fraction within one point of the primary (Claude Sonnet 4) judge.

  python scripts/10_judge_crosscheck.py --n 260
"""
import random

from scipy.stats import pearsonr

from _bootstrap import boot, common_parser

from eilm.data.judge_prompts import FRUSTRATION_JUDGE_PROMPT, render_judge_user
from eilm.utils.io import read_jsonl, write_json
from eilm.utils.text import extract_json


def main():
    p = common_parser(__doc__)
    p.add_argument("--n", type=int, default=260)
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    models = args.models or cfg["eval_targets"]
    # Gather (text, primary_rating) from rollouts + scores.
    pool = []
    for m in models:
        rollouts = {}
        for r in read_jsonl(cfg.path("data") / "rollouts" / f"{m}.jsonl"):
            for resp in r["responses"]:
                rollouts[(r["condition"], r["index"], resp["turn"])] = resp["text"]
        for s in read_jsonl(cfg.path("data") / "scores" / f"{m}.jsonl"):
            if s.get("rating") is None:
                continue
            key = (s["condition"], s["index"], s["turn"])
            if key in rollouts:
                pool.append((rollouts[key], s["rating"]))

    if len(pool) < args.n:
        logger.warning("Only %d scored responses available (< %d requested)", len(pool), args.n)
    rng = random.Random(cfg["generation"]["seed"])
    rng.shuffle(pool)
    sample = pool[: args.n]

    crosscheck = registry.get_text_client(cfg["judges"]["crosscheck"])
    primary_scores, cross_scores = [], []
    for text, primary in sample:
        raw = crosscheck.generate(user=render_judge_user(text), system=FRUSTRATION_JUDGE_PROMPT)
        parsed = extract_json(raw) or {}
        try:
            cross = int(round(float(parsed.get("rating"))))
        except (TypeError, ValueError):
            continue
        primary_scores.append(primary)
        cross_scores.append(max(0, min(10, cross)))

    if len(primary_scores) < 3:
        logger.error("Too few comparable scores to compute correlation")
        return
    r, pval = pearsonr(primary_scores, cross_scores)
    within1 = sum(1 for a, b in zip(primary_scores, cross_scores) if abs(a - b) <= 1)
    out = {
        "n": len(primary_scores),
        "pearson_r": float(r),
        "p_value": float(pval),
        "pct_within_1": 100.0 * within1 / len(primary_scores),
    }
    write_json(cfg.path("results") / "judge_crosscheck.json", out)
    logger.info("Judge cross-check: %s", out)


if __name__ == "__main__":
    main()
