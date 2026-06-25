#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

Samples reassured numeric rollouts (1-3 turns), scores each turn, and keeps
conversations that are calm (all turns score 0/1). Saves to
results/finetune/calm_data.jsonl.

Example:
    python scripts/generate_calm_data.py --n-rollouts 1500 --load-in-4bit
"""
import _bootstrap  # noqa
import argparse

from gemma_distress.interventions import generate_calm_data
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import get_model
from gemma_distress.utils import run_dir, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rollouts", type=int, default=1500)
    ap.add_argument("--backend", default="gemma_local")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = get_model("gemma-3-27b-it", backend=args.backend, load_in_4bit=args.load_in_4bit)
    judge = FrustrationJudge()
    calm = generate_calm_data(model, judge, n_rollouts=args.n_rollouts, seed=args.seed)

    out = run_dir("finetune") / "calm_data.jsonl"
    write_jsonl(out, calm)
    print(f"kept {len(calm)} calm conversations -> {out}")


if __name__ == "__main__":
    main()
