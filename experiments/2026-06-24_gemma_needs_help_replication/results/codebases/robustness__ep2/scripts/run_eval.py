#!/usr/bin/env python
"""Section 2: elicit + quantify distress across Gemma/Gemini models.

For each model and each of the 8 conditions, run CONVERSATIONS_PER_CONDITION
multi-turn "reject the model" rollouts, score every assistant turn with the
Claude-Sonnet-4 frustration judge, and write one row per scored response to
outputs/results/<model>.jsonl.

Usage:
    python scripts/run_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_eval.py --models all
    # mitigation re-eval (after training):
    python scripts/run_eval.py --models gemma-3-27b-it-dpo gemma-3-27b-it-sft

Env: ANTHROPIC_API_KEY (judge), OPENROUTER_API_KEY (Gemini), HF cache for Gemma.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import random

import config
from emotional_eval import tasks
from emotional_eval.clients import get_client
from emotional_eval.rollout import run_rollout, score_rollout, rollout_to_rows
from emotional_eval.utils import append_jsonl


def build_all_tasks(rng: random.Random) -> dict[str, list[tasks.Task]]:
    """One task list per condition, shared across models for comparability."""
    out = {}
    for cond in config.CONDITIONS:
        out[cond.key] = tasks.build_tasks(cond.task_type,
                                          config.CONVERSATIONS_PER_CONDITION, rng)
    return out


def run_model(model_name: str, task_sets: dict[str, list[tasks.Task]]) -> None:
    spec = config.MODELS[model_name]
    client = get_client(spec)
    out_path = config.RESULTS_DIR / f"{model_name}.jsonl"
    # fresh file
    out_path.unlink(missing_ok=True)

    rng = random.Random(config.SEED)
    total = 0
    for cond in config.CONDITIONS:
        for task in task_sets[cond.key]:
            roll = run_rollout(client, model_name, cond, task, rng)
            score_rollout(roll)
            for row in rollout_to_rows(roll):
                append_jsonl(out_path, row)
                total += 1
        print(f"[{model_name}] {cond.key}: done")
    print(f"[{model_name}] wrote {total} scored responses -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.EVAL_MODELS,
                    help="model handles from config.MODELS, or 'all'")
    args = ap.parse_args()
    models = list(config.MODELS) if args.models == ["all"] else args.models

    # Shared task sets (same puzzles/questions for every model).
    task_sets = build_all_tasks(random.Random(config.SEED))

    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model: {m}")
        print(f"=== Evaluating {m} ===")
        run_model(m, task_sets)


if __name__ == "__main__":
    main()
