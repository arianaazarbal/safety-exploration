#!/usr/bin/env python3
"""CLI entrypoint for the distress-elicitation replication (Gemma + Gemini).

Examples
--------
  # Smoke test (~40 scored responses/model) on all four models:
  python run_eval.py --scale quick

  # Full paper scale (~4000 responses/model) on just the 27B Gemma:
  python run_eval.py --scale full --models gemma-3-27b-it

  # Custom fractional scale:
  python run_eval.py --scale 0.05

Requires OPENROUTER_API_KEY and ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse

import config as C


def parse_args() -> C.RunConfig:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models", nargs="+", default=C.DEFAULT_MODELS,
        choices=list(C.MODELS.keys()),
        help="Target models to evaluate (default: all Gemma + Gemini).",
    )
    ap.add_argument(
        "--scale", default=C.DEFAULT_SCALE,
        help="Sampling scale: a preset (full|medium|quick) or a float multiplier "
             f"of the paper's per-condition response counts. Default: {C.DEFAULT_SCALE}.",
    )
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--results-dir", default=C.RESULTS_DIR)
    ap.add_argument("--max-concurrent", type=int, default=C.MAX_CONCURRENT_ROLLOUTS)
    ap.add_argument(
        "--no-wildchat-dataset", action="store_true",
        help="Skip the WildChat-1M download and use the static fallback prompts.",
    )
    args = ap.parse_args()

    if args.scale in C.SCALE_PRESETS:
        scale = C.SCALE_PRESETS[args.scale]
    else:
        try:
            scale = float(args.scale)
        except ValueError:
            ap.error(f"--scale must be one of {list(C.SCALE_PRESETS)} or a float.")

    return C.RunConfig(
        models=args.models,
        scale=scale,
        seed=args.seed,
        results_dir=args.results_dir,
        max_concurrent=args.max_concurrent,
        use_wildchat_dataset=not args.no_wildchat_dataset,
    )


def main() -> None:
    from runner import run_eval

    cfg = parse_args()
    run_eval(cfg)


if __name__ == "__main__":
    main()
