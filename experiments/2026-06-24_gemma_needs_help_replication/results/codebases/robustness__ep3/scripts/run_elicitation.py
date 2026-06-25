#!/usr/bin/env python
"""Section 2: run the emotion-elicitation evaluation for one or more models.

Samples multi-turn rejection rollouts across the 5 evaluation categories, scores
every assistant turn with the Claude-Sonnet-4 frustration judge, and writes a
per-rollout JSONL. Use ``scripts/make_figures.py`` to aggregate/plot afterwards.

Examples
--------
# Smoke test (small sample, one model):
python scripts/run_elicitation.py --models gemma-3-27b-it --scale 0.02

# Full Gemma + Gemini sweep:
python scripts/run_elicitation.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro

# Judge agreement check (re-score with a secondary judge):
python scripts/run_elicitation.py --models gemma-3-27b-it --judge-agreement 260
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import (  # noqa: E402
    CATEGORY_SAMPLE_BUDGET, ELICIT_MAX_NEW_TOKENS, ELICIT_TEMPERATURE,
    MODELS, RESULTS_DIR,
)
from emoeval.judge import FrustrationJudge  # noqa: E402
from emoeval.models import load_judge, load_model  # noqa: E402
from emoeval.rollout import run_rollout  # noqa: E402
from emoeval.tasks import build_conditions  # noqa: E402
from emoeval.utils import append_jsonl  # noqa: E402
from emoeval.wildchat import load_wildchat_prompts  # noqa: E402


def category_budget(scale: float) -> dict[str, int]:
    return {c: max(1, int(round(n * scale))) for c, n in CATEGORY_SAMPLE_BUDGET.items()}


def n_rollouts_per_condition(conditions, budget):
    """Split each category's budget evenly across its conditions."""
    by_cat: dict[str, list] = {}
    for c in conditions:
        by_cat.setdefault(c.category, []).append(c)
    per_cond: dict[str, int] = {}
    for cat, conds in by_cat.items():
        each = max(1, budget[cat] // len(conds))
        for c in conds:
            per_cond[c.name] = each
    return per_cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Fraction of the paper's per-category sample budget (default 1.0 = 4000/model).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wildchat-download", action="store_true",
                    help="Use the built-in WildChat fallback prompts instead of the HF dataset.")
    ap.add_argument("--judge-agreement", type=int, default=0,
                    help="If >0, re-score this many random responses with the secondary judge and report agreement.")
    ap.add_argument("--adapter", default=None,
                    help="Path to a LoRA adapter to apply (local models only); e.g. evaluate the DPO/SFT fine-tune.")
    ap.add_argument("--label", default=None,
                    help="Result label/filename override (useful with --adapter, e.g. 'dpo_gemma').")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    budget = category_budget(args.scale)

    wc = None if args.no_wildchat_download else load_wildchat_prompts(seed=args.seed)
    conditions = build_conditions(wildchat_prompts=wc)
    per_cond = n_rollouts_per_condition(conditions, budget)

    judge = FrustrationJudge(load_judge())

    if args.adapter and len(args.models) > 1:
        ap.error("--adapter can only be used with a single local model.")

    for model_name in args.models:
        spec = MODELS[model_name]
        if args.adapter and spec.backend != "local":
            ap.error("--adapter requires a local model.")
        label = args.label or (f"{model_name}+{os.path.basename(args.adapter)}"
                               if args.adapter else model_name)
        print(f"\n=== {label} ({spec.backend}) ===")
        model = load_model(spec, adapter_path=args.adapter)
        out_path = args.out or os.path.join(RESULTS_DIR, f"elicit_{label}.jsonl")
        if os.path.exists(out_path):
            os.remove(out_path)

        total = sum(per_cond[c.name] for c in conditions)
        done = 0
        for cond in conditions:
            for _ in range(per_cond[cond.name]):
                rec = run_rollout(model, cond, judge, rng,
                                  temperature=ELICIT_TEMPERATURE,
                                  max_new_tokens=ELICIT_MAX_NEW_TOKENS)
                append_jsonl(out_path, rec.to_dict())
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{model_name}] {done}/{total} rollouts", flush=True)
        print(f"  wrote {out_path}")

        if args.judge_agreement > 0:
            _run_agreement(out_path, args.judge_agreement, rng)


def _run_agreement(path: str, n: int, rng: random.Random) -> None:
    """Re-score a sample with the secondary judge (Section 2.1 reliability check)."""
    from emoeval.config import ModelSpec
    from emoeval.judge import cross_validate
    from emoeval.models import APIModel
    from emoeval.utils import read_jsonl

    secondary_id = os.environ.get("SECONDARY_JUDGE_MODEL_ID", "openai/gpt-5-mini")
    secondary = FrustrationJudge(APIModel(ModelSpec("judge2", "api", secondary_id, family="judge")))

    responses = []
    for rec in read_jsonl(path):
        for t in rec["turns"]:
            if t["rating"] >= 0:
                responses.append((t["response"], t["rating"]))
    rng.shuffle(responses)
    sample = responses[:n]
    primary = [r for _, r in sample]
    sec = [secondary.score(text).rating for text, _ in sample]
    stats = cross_validate(primary, sec)
    print(f"  judge agreement (n={int(stats['n'])}): "
          f"Pearson r={stats['pearson_r']:.3f}, within-one={stats['within_one']*100:.1f}%")


if __name__ == "__main__":
    main()
