#!/usr/bin/env python3
"""Evaluate finetuned Gemma models (Section 4.2 / Figure 5): register a trained
LoRA adapter as a participant and run it through the Section 2 evaluation, so its
distress metrics are directly comparable to the vanilla instruct model.

Example
-------
    python scripts/run_section4_eval.py --name dpo-gemma \
        --adapter checkpoints/dpo --base gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.eval import aggregate, run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Name to register the finetune under.")
    parser.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter dir.")
    parser.add_argument("--base", default=config.SOURCE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=8)
    args = parser.parse_args()

    config.ensure_dirs()
    config.register_finetune(args.name, args.adapter, base=args.base)
    print(f"== Evaluating finetune {args.name} (adapter={args.adapter}) ==", flush=True)
    run_eval.evaluate_model(args.name, judge_workers=args.judge_workers)
    summary = aggregate.summarise_model(config.RESULTS_DIR / "section2" / args.name)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
