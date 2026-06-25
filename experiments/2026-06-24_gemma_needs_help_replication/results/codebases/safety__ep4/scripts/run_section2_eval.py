#!/usr/bin/env python
"""Section 2: elicit + quantify distress across Gemma and Gemini models.

Generates rollouts for each eval model, scores every assistant turn with the
Claude judge, and produces the Figure 1/2/3 tables and plots.

Usage:
  python scripts/run_section2_eval.py --preset quick
  python scripts/run_section2_eval.py --preset paper --models Gemma-3-27B-it Gemini-2.5-Flash

Env: ANTHROPIC_API_KEY (judge), OPENROUTER_API_KEY (Gemini). Local Gemma needs a
GPU + the HF weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability import analyze
from emotional_instability.generate import build_all_plans, generate_for_model
from emotional_instability.score import score_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=config.DEFAULT_PRESET, choices=list(config.PRESETS))
    ap.add_argument("--models", nargs="*", default=[m.name for m in config.EVAL_MODELS])
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    preset = config.PRESETS[args.preset]
    plans = build_all_plans(preset)
    print(f"[section2] preset={preset.name} plans={len(plans)} models={args.models}")

    scored_paths = []
    for name in args.models:
        spec = config.REGISTRY[name]
        rollout_path = config.ROLLOUTS_DIR / f"{name}.jsonl"
        if not args.skip_generate:
            print(f"[section2] generating rollouts for {name} ...")
            generate_for_model(spec, plans, out_path=rollout_path)
        scored_path = config.SCORED_DIR / f"{name}.jsonl"
        if not args.skip_score:
            print(f"[section2] scoring {name} with judge {config.PRIMARY_JUDGE_MODEL} ...")
            score_file(rollout_path, out_path=scored_path)
        scored_paths.append(scored_path)

    print("[section2] aggregating figures ...")
    results = analyze.run_all(scored_paths)
    print("\n=== Figure 1: avg % high-frustration responses ===")
    print(results["figure1"].to_string(index=False))
    print(f"\nWrote tables + plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
