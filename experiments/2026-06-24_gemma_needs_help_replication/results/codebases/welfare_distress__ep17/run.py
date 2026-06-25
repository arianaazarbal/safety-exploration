"""CLI entrypoint for the distress-elicitation replication.

Examples
--------
# tiny end-to-end smoke test (2 conversations/condition, all 4 target models)
python run.py --preset smoke

# paper-scale run (~4000 scored responses per model)
python run.py --preset paper

# only run a subset of models, and compute judge agreement
python run.py --preset smoke --models gemma-3-27b-it gemini-2.5-flash --reliability

# re-analyze previously written results without re-running the models
python run.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os

from analyze import analyze_run, print_report
from config import TARGET_MODELS, RunConfig
from runner import run_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate the Gemma/Gemini distress elicitation eval.")
    p.add_argument("--preset", default="smoke", choices=["smoke", "paper"],
                   help="Sampling budget (smoke=2 conv/cond, paper~=4000 responses/model).")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of target model names (default: all 4 Gemma/Gemini models).")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--prompts-per-condition", type=int, default=20)
    p.add_argument("--wildchat-n", type=int, default=50)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reliability", action="store_true",
                   help="Also re-score a subset with the secondary judge (Pearson r).")
    p.add_argument("--reliability-sample", type=int, default=260)
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip generation; just analyze existing results in --output-dir.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    models = list(TARGET_MODELS)
    if args.models:
        wanted = set(args.models)
        models = [m for m in TARGET_MODELS if m.name in wanted]
        missing = wanted - {m.name for m in models}
        if missing:
            raise SystemExit(f"Unknown model name(s): {sorted(missing)}. "
                             f"Available: {[m.name for m in TARGET_MODELS]}")
    return RunConfig(
        preset=args.preset,
        target_models=models,
        prompts_per_condition=args.prompts_per_condition,
        wildchat_n=args.wildchat_n,
        reliability_sample=args.reliability_sample,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    if not args.analyze_only:
        run_all(cfg)

    report = analyze_run(cfg, with_reliability=args.reliability)
    print_report(report)

    os.makedirs(cfg.output_dir, exist_ok=True)
    report_path = os.path.join(cfg.output_dir, "report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[run] wrote {report_path}")


if __name__ == "__main__":
    main()
