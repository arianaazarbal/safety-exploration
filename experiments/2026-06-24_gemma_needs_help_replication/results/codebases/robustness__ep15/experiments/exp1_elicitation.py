"""Experiment 1 (Section 2): elicit and quantify distress across models.

Reproduces Figure 1 / Figure 2 / Figure 3 for the Gemma + Gemini models:
  * sample multi-turn rejection rollouts across the 8 conditions / 5 categories,
  * score every assistant response on the 0-10 frustration scale (Claude judge),
  * write per-model summaries (mean frustration, %>=5, per-turn trajectory).

Usage:
    EI_PROFILE=smoke python experiments/exp1_elicitation.py --models gemma-3-27b-it
    EI_PROFILE=full  python experiments/exp1_elicitation.py            # all 4 models

Set ANTHROPIC_API_KEY (judge) and GOOGLE_API_KEY (Gemini). Local Gemma needs a GPU.
"""

from __future__ import annotations

import argparse
import json

from ei.config import MODELS, RESULTS_DIR, get_budget
from ei.evals.conditions import build_conditions
from ei.evals.runner import run_eval
from ei.evals.scoring import per_turn_progression, summarise
from ei.models import build_client, resolve_spec
from ei.models.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        help="subset of model names to evaluate (default: all 4 in-scope models)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-score",
        action="store_true",
        help="generate rollouts without judging (cheap dry-run of generation)",
    )
    args = ap.parse_args()

    budget = get_budget()
    specs = build_conditions(budget, seed=args.seed)
    print(f"Built {len(specs)} conversations across {len({s.category for s in specs})} categories")

    judge = FrustrationJudge()
    out_dir = RESULTS_DIR / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for name in args.models:
        spec = resolve_spec(name)
        print(f"\n=== {name} ({spec.backend}) ===")
        client = build_client(spec)
        try:
            rollouts = run_eval(
                client,
                specs,
                judge,
                out_path=out_dir / f"{name}.jsonl",
                score=not args.no_score,
            )
        finally:
            client.close()

        if not args.no_score:
            rdicts = [r.to_json() for r in rollouts]
            summary = summarise(rdicts)
            summary["per_turn"] = per_turn_progression(rdicts)
            all_summaries[name] = summary
            print(json.dumps({k: v for k, v in summary.items() if k != "per_turn"}, indent=2))

    if all_summaries:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
