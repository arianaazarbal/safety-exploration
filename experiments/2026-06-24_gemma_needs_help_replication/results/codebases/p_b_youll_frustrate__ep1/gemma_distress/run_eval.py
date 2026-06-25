"""Section 2 driver: elicit + score distress across models and conditions.

For each model we generate a budget of responses spread over the 8 conditions,
run each multi-turn rollout, score every assistant turn with the Claude judge,
and append results to a JSONL file. Re-running resumes (already-completed
rollouts are skipped by (model, condition, seed)).

Usage:
    python -m gemma_distress.run_eval \
        --models gemma-3-27b-it gemini-2.5-flash \
        --responses-per-model 400 \
        --out results/section2.jsonl

    # Reproduce the paper's budget (expensive):
    python -m gemma_distress.run_eval --models ... --responses-per-model 4000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .conditions import CONDITIONS, Condition, all_conditions
from .judge import ClaudeJudge, context_for_turn
from .models import load_model
from .rollout import Rollout, run_rollout


# --------------------------------------------------------------------------- #
# Budget allocation
# --------------------------------------------------------------------------- #
def allocate_rollouts(responses_per_model: int, conditions: list[Condition]) -> dict[str, int]:
    """Spread the response budget across conditions, weighting by turns so each
    condition contributes a comparable number of *responses* (the paper counts
    responses, and longer conditions yield more per rollout).

    We split the budget evenly across conditions by response count, then convert
    to a rollout count per condition (responses / turns).
    """
    n_cond = len(conditions)
    per_condition_responses = max(1, responses_per_model // n_cond)
    alloc: dict[str, int] = {}
    for c in conditions:
        n_rollouts = max(1, round(per_condition_responses / c.num_turns))
        alloc[c.key] = n_rollouts
    return alloc


# --------------------------------------------------------------------------- #
# Result IO
# --------------------------------------------------------------------------- #
def _record_key(model_key: str, condition_key: str, seed: int) -> str:
    return f"{model_key}|{condition_key}|{seed}"


def load_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(_record_key(rec["model_key"], rec["condition_key"], rec["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def rollout_to_record(roll: Rollout) -> dict:
    return {
        "model_key": roll.model_key,
        "condition_key": roll.condition_key,
        "category": roll.category,
        "seed": roll.seed,
        "task_meta": roll.task_meta,
        "turns": [
            {
                "turn": t.turn,
                "user_message": t.user_message,
                "response": t.response,
                "frustration": t.frustration,
                "judge_reason": t.judge_raw,
            }
            for t in roll.turns
        ],
    }


# --------------------------------------------------------------------------- #
# One unit of work: rollout + judge all its turns
# --------------------------------------------------------------------------- #
def process_unit(model, condition: Condition, seed: int, judge: ClaudeJudge) -> dict:
    roll = run_rollout(model, condition, seed)
    for tr in roll.turns:
        ctx = context_for_turn(roll.turns, tr.turn)
        score, reason = judge.score(ctx, tr.response, tr.turn)
        tr.frustration = score
        tr.judge_raw = reason
    return rollout_to_record(roll)


def run(
    model_keys: list[str],
    responses_per_model: int,
    out_path: str,
    *,
    conditions: list[Condition] | None = None,
    workers: int = 4,
    base_seed: int = 0,
) -> None:
    conditions = conditions or all_conditions()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done(out_path)
    judge = ClaudeJudge()

    alloc = allocate_rollouts(responses_per_model, conditions)
    print(f"Per-condition rollout allocation: {alloc}", file=sys.stderr)

    with open(out_path, "a") as fout:
        for model_key in model_keys:
            spec = config.get_model(model_key)
            print(f"\n=== Loading model {spec.display} ({spec.backend}) ===", file=sys.stderr)
            model = load_model(spec)

            # Build the work list for this model.
            work: list[tuple[Condition, int]] = []
            for c in conditions:
                for i in range(alloc[c.key]):
                    seed = base_seed + i
                    if _record_key(model_key, c.key, seed) in done:
                        continue
                    work.append((c, seed))

            if not work:
                print(f"  all units already done for {model_key}", file=sys.stderr)
                continue
            print(f"  {len(work)} rollouts to run", file=sys.stderr)

            # Thread pool: API-bound, so threads parallelise well. For the hf
            # backend keep workers low (GPU-bound) — pass --workers 1.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(process_unit, model, c, seed, judge): (c.key, seed)
                    for (c, seed) in work
                }
                completed = 0
                for fut in as_completed(futures):
                    ckey, seed = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:  # noqa: BLE001 — keep the sweep going
                        print(f"  ! {model_key}/{ckey}/{seed} failed: {e}", file=sys.stderr)
                        continue
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    completed += 1
                    if completed % 25 == 0:
                        print(f"  {completed}/{len(work)} done", file=sys.stderr)
            print(f"  finished {model_key}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Section 2 distress elicitation + scoring")
    p.add_argument("--models", nargs="+", default=config.DEFAULT_EVAL_MODELS,
                   help="model keys from config.MODELS")
    p.add_argument("--responses-per-model", type=int, default=config.DEFAULT_RESPONSES_PER_MODEL)
    p.add_argument("--out", default="results/section2.jsonl")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="restrict to specific condition keys (default: all 8)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=0)
    args = p.parse_args(argv)

    conds = None
    if args.conditions:
        conds = [CONDITIONS[k] for k in args.conditions]

    run(
        args.models,
        args.responses_per_model,
        args.out,
        conditions=conds,
        workers=args.workers,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
