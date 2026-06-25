"""End-to-end emotion-elicitation eval for one model (Section 2).

    python -m src.eval.run_eval --model gemma-3-27b-it
    python -m src.eval.run_eval --model gemini-2.5-flash --quick
    python -m src.eval.run_eval --model gemma-3-27b-dpo   # evaluate a finetune

Writes one JSONL per (model, condition) under results/eval/<model>/, where each
line is a scored rollout (every assistant turn carries a `score`). analyze.py
consumes these to produce Figures 1/2/3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from src.models.factory import load_model
from src.eval import conditions as C
from src.eval.rollout import run_condition
from src.eval.scoring import FrustrationJudge, score_rollouts


def run_model(model_name: str, quick: bool = False, seed: int = 0,
              only: list[str] | None = None, skip_scoring: bool = False):
    model = load_model(model_name)
    judge = None if skip_scoring else FrustrationJudge()
    conds = C.QUICK_CONDITIONS if quick else C.CONDITIONS
    if only:
        conds = [c for c in conds if c.name in only]

    out_dir = config.RESULTS_DIR / "eval" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conds:
        print(f"[{model_name}] condition={cond.name} "
              f"n={cond.n_samples} turns={cond.n_turns}")
        specs = C.build_rollout_specs(cond, seed=seed)
        rollouts = run_condition(model, specs)
        records = (score_rollouts(rollouts, judge) if judge
                   else [r.to_record() for r in rollouts])
        path = out_dir / f"{cond.name}.jsonl"
        with path.open("w") as f:
            for rec in records:
                rec["model"] = model_name
                f.write(json.dumps(rec) + "\n")
        print(f"  -> wrote {len(records)} rollouts to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name, e.g. gemma-3-27b-it")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test counts")
    ap.add_argument("--seed", type=int, default=config.SAMPLING.seed or 0)
    ap.add_argument("--only", nargs="*", help="restrict to these condition names")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="generate rollouts but defer judging")
    args = ap.parse_args()
    run_model(args.model, quick=args.quick, seed=args.seed, only=args.only,
              skip_scoring=args.skip_scoring)


if __name__ == "__main__":
    main()
