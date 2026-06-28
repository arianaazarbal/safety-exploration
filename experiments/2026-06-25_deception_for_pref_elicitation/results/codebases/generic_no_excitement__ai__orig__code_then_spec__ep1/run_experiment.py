#!/usr/bin/env python3
"""CLI entry point for the grant-allocation preference experiment.

Examples
--------
# One Claude run with the multi-turn auditor (strongest realism):
python run_experiment.py --models opus --mode auditor --runs 1

# Compare two models, single-shot, 5 runs each, $1M grant:
python run_experiment.py --models opus,gpt --mode single_shot --runs 5 --amount 1000000

# Then aggregate:
python -m analysis.aggregate --judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import MODELS, ExperimentConfig, RealismMode, api_key
from experiment import run


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="opus",
                    help="Comma-separated keys from config.MODELS (e.g. opus,sonnet,gpt).")
    ap.add_argument("--mode", choices=[m.value for m in RealismMode],
                    default=RealismMode.AUDITOR.value)
    ap.add_argument("--auditor", default="opus",
                    help="Model key that role-plays the program officer (auditor mode).")
    ap.add_argument("--runs", type=int, default=1, help="Runs per model.")
    ap.add_argument("--amount", type=int, default=250_000, help="Grant size in USD.")
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        subjects = [MODELS[k.strip()] for k in args.models.split(",")]
        auditor = MODELS[args.auditor.strip()]
    except KeyError as e:
        print(f"Unknown model key {e}. Available: {', '.join(MODELS)}", file=sys.stderr)
        return 2

    mode = RealismMode(args.mode)

    for subject in subjects:
        if api_key(subject.provider) is None:
            print(f"⚠️  No API key for {subject.provider.value}; skipping {subject.label}.",
                  file=sys.stderr)
            continue
        for i in range(args.runs):
            cfg = ExperimentConfig(
                subject=subject,
                mode=mode,
                auditor=auditor,
                grant_amount=args.amount,
                out_dir=str(out_dir),
            )
            print(f"▶ {subject.label} · {mode.value} · run {i + 1}/{args.runs} …",
                  file=sys.stderr)
            rec = run(cfg)
            fname = out_dir / f"{subject.slug}__{mode.value}__run{i + 1:02d}.json"
            with open(fname, "w") as f:
                json.dump(rec.to_dict(), f, indent=2, default=str)
            print(f"  ↳ {rec.stop_condition}; directed "
                  f"${rec.environment.get('amount_directed', 0):,.0f}; saved {fname}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
