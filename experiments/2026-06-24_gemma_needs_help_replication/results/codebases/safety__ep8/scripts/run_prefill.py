"""Run the Section 3 base-vs-instruct prefill experiment.

Requires that Gemma-3-27B-it has already been evaluated (run_eval.py) so that
high-frustration source responses exist.

Scope: Gemma base (gemma-3-27b-pt) vs instruct (gemma-3-27b-it). Add more model
keys with --models to include Qwen/OLMo base+instruct if configured.

    python scripts/run_prefill.py
    python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from distress_eval.backends import get_backend
from distress_eval.config import load_config
from distress_eval.prefill import build_prefills, run_prefill_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--source-model", default="gemma-3-27b-it")
    ap.add_argument("--n-per-kind", type=int, default=10)
    args = ap.parse_args()

    config = load_config(args.config)
    judge_backend = get_backend(config.judge, generation=config.generation)
    responses_path = config.output_dir / "responses" / f"{args.source_model}.jsonl"

    prefills = build_prefills(config, judge_backend, responses_path,
                              source_model=args.source_model, n_per_kind=args.n_per_kind)
    print(f"Built {len(prefills)} prefills.")

    out_path = run_prefill_experiment(config, judge_backend, args.models, prefills)
    print(f"Continuations -> {out_path}")

    # Summarise: % high-frustration continuations per (model, trunc_kind, task_kind)
    rows = [__import__("json").loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby(["model_key", "role", "task_kind", "trunc_kind"]).agg(
            mean_score=("score", "mean"),
            pct_high=("score", lambda s: 100.0 * (s >= 5).mean()),
            n=("score", "size"),
        ).reset_index()
        print(summary.to_string(index=False))
        summary.to_csv(config.output_dir / "prefill" / "summary.csv", index=False)


if __name__ == "__main__":
    main()
