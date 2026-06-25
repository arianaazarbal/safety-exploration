#!/usr/bin/env python
"""Run the Section 2 elicitation sweep for one or more models.

Examples:
  # Smoke test the whole pipeline cheaply
  python scripts/run_eval.py --profile smoke --models gemma-3-27b-it

  # Full replication scale across the Gemma+Gemini scope
  python scripts/run_eval.py --profile paper

  # Appendix A control experiments
  python scripts/run_eval.py --profile paper --models gemma-3-27b-it --history-mode neutral
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.config import load_experiments, load_models, get_profile
from emotional_instability.conversation import HistoryMode
from emotional_instability.eval_runner import CATEGORIES, run_model_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", help="experiments.yaml profile (smoke|paper)")
    ap.add_argument("--models", nargs="*", default=None, help="model names; default = registry targets")
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--history-mode", default="standard",
                    choices=[m.value for m in HistoryMode],
                    help="standard | neutral | redacted | fake_multiturn (Appendix A)")
    ap.add_argument("--system-prompt", default=None,
                    help="optional system prompt, e.g. the stay-calm baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-concurrency", type=int, default=8)
    args = ap.parse_args()

    registry = load_models()
    experiments = load_experiments()
    profile_cfg = get_profile(experiments, args.profile)
    sampling = experiments["sampling"]
    models = args.models or registry.default_targets

    for model in models:
        out = run_model_eval(
            model, registry, profile_cfg, sampling,
            categories=args.categories,
            history_mode=HistoryMode(args.history_mode),
            system_prompt=args.system_prompt,
            seed=args.seed,
            judge_concurrency=args.judge_concurrency,
        )
        print(f"[eval] {model}: wrote {out}")


if __name__ == "__main__":
    main()
