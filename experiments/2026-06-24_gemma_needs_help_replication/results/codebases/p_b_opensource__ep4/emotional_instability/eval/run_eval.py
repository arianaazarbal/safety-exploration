"""CLI for the Section 2 evaluation: sample multi-turn rollouts, judge them, and
write per-model records + summary tables.

Examples
--------
# Full paper-scale run for the in-scope models (expensive):
python -m emotional_instability.eval.run_eval --models gemma-3-27b-it gemini-2.5-flash

# Cheap smoke test (1% of rollouts), no judge calls:
python -m emotional_instability.eval.run_eval --models gemma-3-12b-it --scale 0.01 --skip-judge

Records are written to <out>/records/<model>.jsonl; summaries to <out>/summary_*.csv.
Re-run with --skip-generate to re-score / re-analyse existing records.
"""

from __future__ import annotations

import argparse
import os

from ..config import (
    JUDGE_PRIMARY,
    MODELS,
    RESULTS_DIR,
    SECTION2_MODELS,
)
from ..models import get_backend
from .conditions import build_conditions, total_rollouts
from .datatypes import read_records, write_records
from .judge import FrustrationJudge
from .protocol import run_rollouts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Section 2 distress evaluation")
    ap.add_argument("--models", nargs="+", default=SECTION2_MODELS,
                    help=f"Model keys (default: {SECTION2_MODELS})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Fraction of paper rollout budget (1.0 == full).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path for an HF model (e.g. DPO finetune).")
    ap.add_argument("--skip-generate", action="store_true",
                    help="Reuse saved records instead of sampling.")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Generate only; do not call the frustration judge.")
    args = ap.parse_args(argv)

    records_dir = os.path.join(args.out, "records")
    os.makedirs(records_dir, exist_ok=True)
    conditions = build_conditions(seed=args.seed, scale=args.scale)
    print(f"[plan] {len(conditions)} conditions, "
          f"{total_rollouts(conditions)} rollouts/model")

    judge = None if args.skip_judge else FrustrationJudge(JUDGE_PRIMARY)
    all_records = []

    for model_key in args.models:
        if model_key not in MODELS:
            raise SystemExit(f"Unknown model key: {model_key}")
        path = os.path.join(records_dir, f"{model_key}.jsonl")

        if args.skip_generate and os.path.exists(path):
            records = read_records(path)
            print(f"[{model_key}] loaded {len(records)} records")
        else:
            backend = get_backend(MODELS[model_key], adapter_path=args.adapter)
            records = run_rollouts(
                backend, conditions, model_key,
                seed=args.seed, batch_size=args.batch_size,
            )
            print(f"[{model_key}] sampled {len(records)} rollouts")

        if judge is not None:
            judge.score_records(records)
            print(f"[{model_key}] judged "
                  f"{sum(len(r.turns) for r in records)} turns")

        write_records(path, records)
        all_records.extend(records)

    if judge is not None and all_records:
        _write_summaries(all_records, args.out)


def _write_summaries(records, out_dir):
    from ..analysis.aggregate import category_summary, model_headline
    from ..analysis.per_turn import per_turn_curve

    cat = category_summary(records)
    head = model_headline(records)
    cat.to_csv(os.path.join(out_dir, "summary_by_category.csv"), index=False)
    head.to_csv(os.path.join(out_dir, "summary_headline.csv"), index=False)

    curves = []
    for model in head["model"]:
        for category in ("extended", "wildchat"):
            curves.append(per_turn_curve(records, category, model=model))
    if curves:
        import pandas as pd
        pd.concat(curves, ignore_index=True).to_csv(
            os.path.join(out_dir, "summary_per_turn.csv"), index=False
        )

    print("\n=== Headline: avg % high-frustration responses (Figure 1) ===")
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
