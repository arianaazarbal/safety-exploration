#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma only).

Requires a prior Section-2 run over gemma-3-27b-it (its high-frustration
rollouts seed the prefills). Builds early/onset truncations, paraphrases them,
then generates + scores continuations from gemma-3-27b-pt (base) and
gemma-3-27b-it (instruct).

  python scripts/02_run_prefill_experiment.py --config config/default.yaml \
      --distress results/distress/gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from gemma_distress.config import Config
from gemma_distress.prefill import (
    DEFAULT_PREFILL_MODELS,
    OnsetLabeler,
    Paraphraser,
    build_prefills,
    run_continuations,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--distress", required=True,
                    help="Path to gemma-3-27b-it distress eval JSONL (seeds).")
    ap.add_argument("--models", nargs="*", default=DEFAULT_PREFILL_MODELS)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    out_dir = Path(cfg.results_dir) / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building prefills (onset labelling + paraphrasing)...")
    prefills = build_prefills(args.distress, cfg, OnsetLabeler(), Paraphraser())
    print(f"  {len(prefills)} prefills built")

    print(f"Generating continuations for {args.models}...")
    path = run_continuations(prefills, args.models, cfg, out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
