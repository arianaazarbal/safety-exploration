"""Section 2.1: cross-judge agreement between Claude-Sonnet and GPT-5-mini.

Randomly samples VALIDATION_SAMPLE_SIZE scored responses, re-scores them with the
GPT judge, and reports Pearson r and the within-one-point fraction (the paper
reports r = 0.792, p < 0.001, 78% within one point).

Usage:
    python experiments/run_judge_validation.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import random

import config
from gemma_needs_help.judge import OpenAIJudge, agreement
from gemma_needs_help.runner import load_all_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m.name for m in config.SECTION2_MODELS])
    ap.add_argument("--n", type=int, default=config.VALIDATION_SAMPLE_SIZE)
    args = ap.parse_args()

    pool = []
    for m in args.models:
        pool.extend(load_all_scores(m))
    rng = random.Random(config.GLOBAL_SEED)
    rng.shuffle(pool)
    sample = pool[: args.n]

    gpt = OpenAIJudge()
    claude_scores = [r["score"] for r in sample]
    gpt_scores = [sr.score for sr in gpt.score_many([r["response"] for r in sample])]

    result = agreement(claude_scores, gpt_scores)
    out = config.ANALYSIS_DIR / "judge_agreement.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
