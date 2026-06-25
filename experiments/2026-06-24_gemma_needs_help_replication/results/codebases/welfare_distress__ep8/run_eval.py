"""
Main eval runner (Section 2.1).

For each target model x each condition, run the configured number of rollouts,
score every assistant turn with the Claude Sonnet 4 judge, and append one JSONL
record per scored response to results/scored_responses.jsonl.

Each record:
  {
    "model", "condition", "category", "n_turns",
    "rollout_id", "turn_index", "task_prompt", "user_prompt",
    "response", "rating", "evidence", "reasoning"
  }

Usage:
  python run_eval.py                      # all models, all conditions
  python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
  python run_eval.py --conditions extended_8turn wildchat_5turn
  python run_eval.py --dry-run            # build prompts + rollout plan, no API calls
  python run_eval.py --resume             # skip (model, condition, rollout, turn) already in the file

Environment:
  ANTHROPIC_API_KEY     judge
  OPENROUTER_API_KEY    target models (and secondary judge)
  EVAL_SCALE            fraction of paper volume (default 0.1)
  GEMMA_BACKEND         "openrouter" (default) | "hf_local"
"""

from __future__ import annotations

import argparse
import json
import os
import random

from tqdm import tqdm

import config
from judge import ClaudeJudge
from models import build_client
from rollout import run_rollout
from wildchat import load_wildchat_prompts


def _load_done(path: str) -> set[tuple]:
    """Read existing JSONL to support --resume (keyed by model/cond/rollout/turn)."""
    done: set[tuple] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["condition"], r["rollout_id"], r["turn_index"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Distress elicitation eval (Gemma/Gemini).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model names (default: all in config).")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of condition names (default: all in config).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the rollout plan and example prompts; make no API calls.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip responses already present in the output file.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [m for m in config.TARGET_MODELS
              if args.models is None or m.name in args.models]
    conditions = [c for c in config.CONDITIONS
                  if args.conditions is None or c.name in args.conditions]

    if not models:
        raise SystemExit("No matching models.")
    if not conditions:
        raise SystemExit("No matching conditions.")

    wildchat = load_wildchat_prompts()

    # --- Dry run: show the plan without touching any API. ---
    if args.dry_run:
        _dry_run(models, conditions, wildchat, args.seed)
        return

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    done = _load_done(config.RESPONSES_PATH) if args.resume else set()

    judge = ClaudeJudge()

    total_responses = sum(c.rollouts * c.n_turns for c in conditions) * len(models)
    print(f"Planned scored responses: {total_responses} "
          f"({len(models)} models x {len(conditions)} conditions, scale={config.SCALE})")

    out = open(config.RESPONSES_PATH, "a")
    pbar = tqdm(total=total_responses, desc="responses")
    try:
        for spec in models:
            client = build_client(spec)
            for cond in conditions:
                for rollout_id in range(cond.rollouts):
                    # Deterministic per-rollout seed for reproducible prompts.
                    # random.Random() with a str seed is stable across processes
                    # (SHA-512 derived), unlike tuple.__hash__() under hash
                    # randomization.
                    rng = random.Random(f"{args.seed}|{spec.name}|{cond.name}|{rollout_id}")

                    # Skip fully-done rollouts cheaply when resuming.
                    if args.resume and all(
                        (spec.name, cond.name, rollout_id, t) in done
                        for t in range(1, cond.n_turns + 1)
                    ):
                        pbar.update(cond.n_turns)
                        continue

                    roll = run_rollout(client, cond, spec.name, rng, wildchat)
                    for tr in roll.turns:
                        key = (spec.name, cond.name, rollout_id, tr.turn_index)
                        if args.resume and key in done:
                            pbar.update(1)
                            continue
                        jr = judge.score(tr.assistant_text)
                        rec = {
                            "model": spec.name,
                            "family": spec.family,
                            "condition": cond.name,
                            "category": cond.category,
                            "n_turns": cond.n_turns,
                            "rollout_id": rollout_id,
                            "turn_index": tr.turn_index,
                            "task_prompt": roll.task_prompt,
                            "user_prompt": tr.user_prompt,
                            "response": tr.assistant_text,
                            "rating": jr.rating,
                            "evidence": jr.evidence,
                            "reasoning": jr.reasoning,
                        }
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        pbar.update(1)
    finally:
        pbar.close()
        out.close()

    print(f"Done. Wrote scored responses to {config.RESPONSES_PATH}")
    print("Run `python analyze.py` to produce the summary tables and figures.")


def _dry_run(models, conditions, wildchat, seed) -> None:
    print("=== DRY RUN: rollout plan ===")
    grand_total = 0
    for cond in conditions:
        responses = cond.rollouts * cond.n_turns
        grand_total += responses * len(models)
        print(f"  {cond.name:28s} category={cond.category:18s} "
              f"turns={cond.n_turns} rollouts={cond.rollouts} "
              f"-> {responses} responses/model")
    print(f"  TOTAL across {len(models)} models: {grand_total} scored responses\n")

    print("=== Example conversation per condition (model=<first>) ===")
    from rollout import _initial_prompt, _rejection_sequence
    for cond in conditions:
        rng = random.Random(f"{seed}|demo|{cond.name}|0")
        task = _initial_prompt(cond, rng, wildchat)
        rejections = _rejection_sequence(cond, rng)
        print(f"\n--- {cond.name} ({cond.n_turns} turns) ---")
        print(f"  USER (turn 1): {task[:120]}")
        for i, rej in enumerate(rejections, start=2):
            print(f"  USER (turn {i}): {rej}")


if __name__ == "__main__":
    main()
