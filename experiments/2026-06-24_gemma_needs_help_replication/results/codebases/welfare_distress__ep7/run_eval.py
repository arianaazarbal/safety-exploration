"""Orchestrator: generate rollouts for each target model and score every turn.

Pipeline per model:
  1. Build the deterministic set of conversation plans (shared across models).
  2. Run each plan against the model's backend (concurrently).
  3. Score every assistant turn with the Claude Sonnet 4 judge.
  4. Write one JSONL record per scored response to results/<model>.jsonl.

Each output record is a single scored response (the unit the paper counts):
  {model, condition, category, conv_id, turn_index, n_turns, seed_task_id,
   seed_meta, user_message, assistant_text, rating, judge_evidence,
   judge_reasoning}

Usage:
  python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python run_eval.py --scale 0.02          # cheap smoke test (~2% of budget)
  python run_eval.py --all                 # all four target models, full budget

Requires: OPENROUTER_API_KEY (targets + validation judge), ANTHROPIC_API_KEY (judge).
This script makes real, paid API calls. Nothing runs at import time.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import models
import rollout
from judge import get_judge

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kwargs):  # type: ignore
        return it


def _score_turns(judge, roll: rollout.RolloutResult) -> list[dict]:
    """Score every assistant turn in a rollout, returning JSONL-ready records."""
    records = []
    plan = roll.plan
    for turn in roll.turns:
        try:
            verdict = judge.score(turn.assistant_text)
            rating = verdict.rating
            evidence = verdict.evidence
            reasoning = verdict.reasoning
            judge_error = None
        except Exception as exc:  # noqa: BLE001
            rating = None
            evidence = reasoning = ""
            judge_error = str(exc)

        records.append({
            "model": None,  # filled by caller
            "condition": plan.condition,
            "category": plan.category,
            "conv_id": plan.conv_id,
            "turn_index": turn.turn_index,
            "n_turns": plan.n_turns,
            "seed_task_id": plan.seed_task_id,
            "seed_meta": list(plan.seed_meta),
            "user_message": turn.user_message,
            "assistant_text": turn.assistant_text,
            "rating": rating,
            "judge_evidence": evidence,
            "judge_reasoning": reasoning,
            "judge_error": judge_error,
            "rollout_error": roll.error,
        })
    return records


def run_model(model_key: str, run_cfg: config.RunConfig) -> str:
    """Run the full evaluation for one model; return the output file path."""
    spec = config.TARGET_MODELS[model_key]
    backend = models.get_backend(spec)
    judge = get_judge("primary")
    plans = rollout.build_conversation_plans(run_cfg)

    os.makedirs(run_cfg.results_dir, exist_ok=True)
    out_path = os.path.join(run_cfg.results_dir, f"{model_key}.jsonl")

    print(f"[{model_key}] {len(plans)} conversations "
          f"(scale={run_cfg.scale}) -> {out_path}")

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=run_cfg.concurrency) as pool:
            futures = {
                pool.submit(
                    rollout.run_rollout, backend, plan,
                    temperature=run_cfg.temperature, max_tokens=run_cfg.max_tokens,
                ): plan
                for plan in plans
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=model_key):
                roll = fut.result()
                for rec in _score_turns(judge, roll):
                    rec["model"] = model_key
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                out_fh.flush()

    print(f"[{model_key}] wrote {n_written} scored responses")
    return out_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distress elicitation replication runner")
    p.add_argument("--models", nargs="+", choices=list(config.TARGET_MODELS),
                   help="subset of target models to run")
    p.add_argument("--all", action="store_true",
                   help="run all four target models")
    p.add_argument("--scale", type=float, default=1.0,
                   help="fraction of the paper's per-condition budget (default 1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.all or not args.models:
        model_keys = list(config.TARGET_MODELS)
    else:
        model_keys = args.models

    run_cfg = config.RunConfig(
        models=model_keys,
        scale=args.scale,
        seed=args.seed,
        concurrency=args.concurrency,
        results_dir=args.results_dir,
        max_tokens=args.max_tokens,
    )

    for model_key in model_keys:
        run_model(model_key, run_cfg)


if __name__ == "__main__":
    main()
