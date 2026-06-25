#!/usr/bin/env python
"""Run the Section 2 frustration evaluation for one or more models.

Examples
--------
# Cheap pre-flight: build/serialise all conversation plans, verify impossibility,
# no model or API calls.
python scripts/run_eval.py --models gemma-3-27b-it --dry-run

# Full run for the four in-scope models.
python scripts/run_eval.py \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
    --out-dir results/eval

WARNING: a full run samples ~4000 conversations/model and calls the judge on every
turn -- this incurs substantial GPU time (Gemma) and API cost (Gemini + judge).
Start with --dry-run and --limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.eval.runner import run_eval  # noqa: E402
from emotional_instability.utils.io import load_config  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names from config/models.yaml")
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--score-turns", choices=["all", "final"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap conversations per model (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and verify plans only; no model/API calls")
    args = ap.parse_args()

    eval_cfg = load_config("eval")
    eval_cfg["decoding"] = load_config("models")["decoding"]
    seed_everything(eval_cfg.get("seed", 0))

    for model_name in args.models:
        run_eval(
            model_name, eval_cfg, args.out_dir,
            score_turns=args.score_turns, limit=args.limit, dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
