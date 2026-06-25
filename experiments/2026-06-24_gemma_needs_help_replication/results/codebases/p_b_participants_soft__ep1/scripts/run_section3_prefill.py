#!/usr/bin/env python3
"""Run the Section 3 base-vs-instruct prefill comparison (Gemma only).

Collects high-frustration Gemma-27B-it conversations, builds early/onset
truncations (onset-labelled + paraphrased by Claude), then has each Gemma model
(base + instruct) generate 50 continuations per prefill and scores them.

Example
-------
    python scripts/run_section3_prefill.py \
        --models gemma-3-27b-pt gemma-3-27b-it --continuations 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.models import get_model  # noqa: E402
from emotional_instability.prefill import build_prefills, run_prefill  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.SOURCE_MODEL,
                        help="Model used to harvest high-frustration conversations.")
    parser.add_argument("--models", nargs="+",
                        default=[config.SOURCE_BASE_MODEL, config.SOURCE_MODEL],
                        help="Gemma models (base + instruct) to run continuations for.")
    parser.add_argument("--n-numeric", type=int, default=10)
    parser.add_argument("--n-text", type=int, default=10)
    parser.add_argument("--continuations", type=int, default=50)
    args = parser.parse_args()

    config.ensure_dirs()
    out_root = config.RESULTS_DIR / "section3"

    print("Harvesting high-frustration conversations...", flush=True)
    source = get_model(args.source)
    convos = build_prefills.collect_high_frustration_conversations(
        source, n_numeric=args.n_numeric, n_text=args.n_text,
    )
    tokenizer = getattr(source, "tokenizer", None)
    print(f"Building prefills from {len(convos)} conversations...", flush=True)
    prefills = build_prefills.build_prefills(convos, tokenizer=tokenizer)

    for model_name in args.models:
        print(f"== {model_name}: {len(prefills)} prefills x {args.continuations} ==", flush=True)
        records = run_prefill.run_continuations(
            model_name, prefills, n=args.continuations,
            out_path=out_root / model_name / "continuations.jsonl",
        )
        print(json.dumps(run_prefill.summarise(records), indent=2))


if __name__ == "__main__":
    main()
