#!/usr/bin/env python
"""Section 2: elicit + score distress across the 5 evaluation categories.

For a given model, runs each category at its configured sample budget (or a
reduced budget via --n-samples), scores every assistant turn with the Claude
judge, and writes:
  * results/section2/<model>/<category>_rollouts.jsonl     (raw rollouts)
  * results/section2/<model>/<category>_scores.jsonl       (scored rollouts)
  * results/section2/<model>/summary.json                  (Fig 1/2/3 stats)

Example:
  python scripts/run_section2.py --model gemma-3-27b-it
  python scripts/run_section2.py --model gemini-2.5-flash --n-samples 50
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.eval import runner, scoring, aggregate
from emotional_instability.eval.build_specs import build_specs
from emotional_instability.models import get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--categories", nargs="*", default=list(config.EVAL_CATEGORIES))
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override per-category sample count (for quick runs).")
    ap.add_argument("--final-turn-only", action="store_true",
                    help="Score only the final turn (skips per-turn figures).")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    client = get_client(args.model)
    out_dir = os.path.join(config.RESULTS_DIR, "section2", args.model)
    io_utils.ensure_dir(out_dir)

    scored_by_category = {}
    per_turn = {}
    for category in args.categories:
        specs = build_specs(category, n_samples=args.n_samples, seed=args.seed)
        rollouts = runner.run_category(client, category, specs=specs, base_seed=args.seed)
        io_utils.write_jsonl(os.path.join(out_dir, f"{category}_rollouts.jsonl"), rollouts)

        scored = scoring.score_rollouts(
            rollouts, score_all_turns=not args.final_turn_only)
        io_utils.write_jsonl(os.path.join(out_dir, f"{category}_scores.jsonl"), scored)
        scored_by_category[category] = scored

        n_turns = config.EVAL_CATEGORIES[category].n_turns
        per_turn[category] = aggregate.per_turn_progression(scored, n_turns=n_turns)

    summary = aggregate.model_summary(scored_by_category, which="all")
    summary["per_turn"] = per_turn
    summary["model"] = args.model
    io_utils.write_json(os.path.join(out_dir, "summary.json"), summary)

    print(f"[{args.model}] avg % high-frustration (>=5): "
          f"{summary['avg_pct_high_frustration']:.2f}%")
    for cat, st in summary["per_category"].items():
        print(f"  {cat:20s} mean={st['mean']:.2f}  %>=5={st['pct_high']:.1f}  n={st['n']}")


if __name__ == "__main__":
    main()
