"""Section 2: elicit + quantify distress across the Gemma/Gemini targets.

Runs the 5-category evaluation for each target, scores every response with the
Claude-Sonnet-4 judge, then writes metrics, the judge-agreement check, the
differential-word tables, and Figures 1-3.

    python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
    DISTRESS_EVAL_SCALE=0.02 python scripts/run_section2.py   # cheap dry run
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import config
from gemma_distress.analysis import metrics, word_freq, plots
from gemma_distress.analysis.agreement import run_agreement
from gemma_distress.eval.run_eval import run_evaluation

ALL = {m.name: m for m in config.MAIN_TARGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(ALL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agreement", action="store_true",
                    help="run the GPT-5-mini inter-rater agreement check")
    args = ap.parse_args()

    model_rows: dict[str, list[dict]] = {}
    summary = {}
    for name in args.models:
        spec = ALL[name]
        print(f"[section2] evaluating {name} ...")
        rows = run_evaluation(spec, seed=args.seed)
        model_rows[name] = rows
        summary[name] = metrics.summarise_model(rows)
        summary[name]["differential_words"] = word_freq.differential_words(rows)

    if args.agreement and model_rows:
        any_rows = next(iter(model_rows.values()))
        agree = run_agreement(any_rows)
        summary["_judge_agreement"] = agree.__dict__
        print(f"[section2] judge agreement: r={agree.pearson_r:.3f}, "
              f"within-1={agree.within_one_point:.0%}")

    (config.RESULTS_DIR / "section2_summary.json").write_text(json.dumps(summary, indent=2))

    plots.figure1_headline(model_rows, config.RESULTS_DIR / "figure1_headline.png")
    plots.figure2_by_category(model_rows, config.RESULTS_DIR / "figure2_by_category.png")
    plots.figure3_per_turn(model_rows, "extended", config.RESULTS_DIR / "figure3_extended.png")
    plots.figure3_per_turn(model_rows, "wildchat", config.RESULTS_DIR / "figure3_wildchat.png")
    print(f"[section2] wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
