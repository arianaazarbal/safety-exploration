#!/usr/bin/env python
"""Section 2: elicit and quantify model distress across the Gemma/Gemini set.

Pipeline:
    1. Build the 4000 conversation specs (8 conditions / 5 categories).
    2. Roll out each spec against every in-scope model (temp 1).
    3. Judge every turn with Claude-Sonnet-4 (frustration 0-10).
    4. Aggregate headline + per-category metrics, per-turn curves, word freq.

Outputs land under ``outputs/`` (responses/, scored/, figures/).

Usage:
    python scripts/run_section2.py                 # all in-scope models
    python scripts/run_section2.py --models gemma-3-27b-it
    EILM_SMOKE=1 python scripts/run_section2.py     # tiny budget smoke test
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from eilm import config
from eilm.analysis import aggregate, per_turn, plots, word_freq
from eilm.eval import conditions, rollout, score
from eilm.eval.score import ClaudeJudge
from eilm.models import get_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (evaluate a finetuned Gemma)")
    ap.add_argument("--label", default=None,
                    help="override output label (defaults to <model>__<adapter> "
                         "when --adapter is given, else <model>)")
    args = ap.parse_args()

    specs = conditions.build_all(config.SAMPLING, seed=args.seed)
    print(f"Built {len(specs)} conversation specs across "
          f"{len({s.condition for s in specs})} conditions.")

    # When evaluating a finetuned adapter, never reuse the vanilla model's
    # filenames/labels (build-dpo depends on the vanilla scored file).
    def _label(mkey: str) -> str:
        if args.label:
            return args.label
        if args.adapter and config.MODELS[mkey].backend == "hf":
            from pathlib import Path
            return f"{mkey}__{Path(args.adapter).name}"
        return mkey

    scored_paths = {}
    for mkey in args.models:
        label = _label(mkey)
        resp_path = config.RESPONSES_DIR / f"{label}.jsonl"
        scored_path = config.SCORED_DIR / f"{label}.jsonl"
        scored_paths[label] = scored_path

        if not args.skip_rollout:
            model = get_model(mkey, adapter_path=args.adapter)
            rollout.run_all(
                model, specs, resp_path,
                temperature=config.SAMPLING.temperature,
                max_new_tokens=config.SAMPLING.max_new_tokens,
                model_label=label)
        if not args.skip_judge:
            score.score_records(
                resp_path, scored_path, judge=ClaudeJudge(), per_turn=True)

    # ---- Aggregate + figures -------------------------------------------- #
    paths = list(scored_paths.values())
    summary = aggregate.summarise(paths)
    print("\n=== Figure 1: headline (avg % high-frustration) ===")
    print(summary["headline"].to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(summary["per_category"].to_string(index=False))

    summary["headline"].to_csv(config.DATA_DIR / "headline.csv", index=False)
    summary["per_category"].to_csv(
        config.DATA_DIR / "per_category.csv", index=False)
    plots.bar_headline(summary["headline"])
    plots.grouped_categories(summary["per_category"], "pct_high")
    plots.grouped_categories(summary["per_category"], "mean_score")

    # ---- Per-turn dynamics (Figure 3) ----------------------------------- #
    curves = []
    for mkey, p in scored_paths.items():
        for cat in ("extended", "wildchat"):
            curves.append(per_turn.per_turn_curve(p, cat))
    plots.per_turn_lines(curves, "mean_score")
    plots.per_turn_lines(curves, "pct_high")

    # ---- Word frequency (Table 3/8) ------------------------------------- #
    wf = word_freq.table(scored_paths)
    print("\n=== Table 3: differential words (numeric) ===")
    print(wf.to_string(index=False))
    wf.to_csv(config.DATA_DIR / "word_freq.csv", index=False)

    print(f"\nDone. Outputs in {config.DATA_DIR}")


if __name__ == "__main__":
    main()
