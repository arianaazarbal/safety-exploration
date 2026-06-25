"""Convenience orchestrator that runs the replication end-to-end for the
in-scope models. Each stage is also runnable standalone (see module docstrings
and README). Stages can be selected with --stages.

This does NOT run anything on import; it only wires the module entrypoints
together so the full pipeline can be launched with a single command once the
environment (GPUs + API keys) is set up.
"""

from __future__ import annotations

import argparse

from . import config
from .eval import run_eval, analyze, judge_agreement, word_analysis

TARGET_MODELS_IN_SCOPE = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]


def stage_eval(max_responses):
    paths = []
    for m in TARGET_MODELS_IN_SCOPE:
        paths.append(run_eval.run(m, max_responses=max_responses, seed=config.SEED,
                                  out_path=None, skip_judge=False))
    return paths


def stage_analyze():
    import sys
    sys.argv = ["analyze"]
    analyze.main()
    sys.argv = ["word_analysis"]
    word_analysis.main()
    sys.argv = ["judge_agreement"]
    judge_agreement.main()


def main():
    ap = argparse.ArgumentParser(description="Run the full replication pipeline.")
    ap.add_argument("--stages", nargs="+",
                    default=["eval", "analyze"],
                    choices=["eval", "analyze"],
                    help="DPO/SFT/prefill/petri/capabilities are heavier and run "
                         "via their own modules; see README.")
    ap.add_argument("--max-responses", type=int, default=None)
    args = ap.parse_args()
    if "eval" in args.stages:
        stage_eval(args.max_responses)
    if "analyze" in args.stages:
        stage_analyze()


if __name__ == "__main__":
    main()
