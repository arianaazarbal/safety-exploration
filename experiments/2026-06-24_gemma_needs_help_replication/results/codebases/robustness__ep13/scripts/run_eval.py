#!/usr/bin/env python
"""Run the Section 2 frustration evaluation for one or more models.

For each model in the config:
  1. generate multi-turn rejection rollouts across the 8 conditions,
  2. score every assistant turn with the judge,
  3. write a per-model metrics summary (mean score, %>=5, per-turn, headline).

Usage:
    python scripts/run_eval.py --config configs/smoke.yaml
    python scripts/run_eval.py --config configs/full.yaml --only gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emotional_instability.config import ExperimentConfig
from emotional_instability.eval import metrics as M
from emotional_instability.eval.judge_runner import score_conversation_file
from emotional_instability.eval.runner import run_all
from emotional_instability.models import build_backend
from emotional_instability.prompts.conditions import scaled_conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="run a single model by name/preset")
    ap.add_argument("--skip-judge", action="store_true", help="generate rollouts only")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    conditions = scaled_conditions(cfg.scale)
    os.makedirs(cfg.output_dir, exist_ok=True)

    judge = None if args.skip_judge else build_backend(cfg.judge)

    summaries = []
    for spec in cfg.models:
        model_name = spec.get("preset") or spec.get("name") or spec.get("model_id")
        if args.only and args.only != model_name:
            continue
        print(f"=== Model: {model_name} ===", flush=True)
        backend = build_backend(spec)

        conv_path = os.path.join(cfg.output_dir, f"{model_name}.conversations.jsonl")
        run_all(
            backend,
            conditions,
            conv_path,
            seed=cfg.seed,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        print(f"  rollouts -> {conv_path}", flush=True)

        if judge is None:
            continue
        scored_path = os.path.join(cfg.output_dir, f"{model_name}.scored.jsonl")
        score_conversation_file(judge, conv_path, scored_path)
        print(f"  scores   -> {scored_path}", flush=True)

        df = M.load_scores(scored_path)
        summary = M.summarise_model(df)
        per_turn = M.per_turn_metrics(df).to_dict(orient="records")
        summary["per_turn"] = per_turn
        sum_path = os.path.join(cfg.output_dir, f"{model_name}.summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
        print(
            f"  summary  -> {sum_path}  "
            f"(headline %>=5 = {summary['headline_pct_high']:.1f}, "
            f"overall mean = {summary['overall_mean']:.2f})",
            flush=True,
        )

    if summaries:
        with open(os.path.join(cfg.output_dir, "summary_all.json"), "w") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
