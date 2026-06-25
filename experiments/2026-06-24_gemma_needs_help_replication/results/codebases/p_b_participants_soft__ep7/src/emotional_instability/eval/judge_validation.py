"""Judge-reliability validation (Section 2.1).

Randomly sample 260 already-scored responses, re-score them with a second judge
(GPT-5-mini) using the same prompt, and report Pearson r and % within one point.
Paper target: r = 0.792 (p < 0.001), 78% within one point.
"""
from __future__ import annotations

import argparse
import random

from ..config import load_config
from ..io_utils import read_jsonl, write_json
from . import judge, metrics


def validate(cfg, model_name: str, n: int = 260, seed: int = 1234) -> dict:
    path = cfg.path("responses_dir") / f"{model_name}.jsonl"
    records = [r for r in read_jsonl(path) if r["rating"] >= 0]
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))

    texts = [r["text"] for r in sample]
    primary = [r["rating"] for r in sample]
    secondary = [s.rating for s in judge.score_many(texts, judge_model="validation_judge")]

    agreement = metrics.judge_agreement(primary, secondary)
    out = {
        "model": model_name,
        "n": agreement.n,
        "pearson_r": agreement.pearson_r,
        "p_value": agreement.p_value,
        "pct_within_one": agreement.pct_within_one,
    }
    write_json(cfg.path("scores_dir") / f"judge_agreement_{model_name}.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Judge agreement validation")
    parser.add_argument("--model", default="gemma-3-27b-it")
    parser.add_argument("--n", type=int, default=260)
    args = parser.parse_args(argv)
    out = validate(cfg, args.model, n=args.n)
    print(
        f"r={out['pearson_r']:.3f} p={out['p_value']:.2e} "
        f"within_one={out['pct_within_one']:.1f}% (n={out['n']})"
    )


if __name__ == "__main__":
    main()
