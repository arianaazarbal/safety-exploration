"""Orchestration entry point.

Builds the full conversation grid for the in-scope models (Gemma + Gemini),
runs each conversation (sequential within a conversation, parallel across
conversations), scores every assistant turn with the primary judge, and writes
results to a JSONL file per model. Optionally re-scores a random subset with the
cross-check judge to reproduce inter-judge agreement.

Usage:
  python -m distress_eval.run_eval                # all in-scope models, full budget
  python -m distress_eval.run_eval --models gemma-3-27b-it gemini-2.5-flash
  EVAL_SCALE=0.02 python -m distress_eval.run_eval   # tiny smoke run
  python -m distress_eval.run_eval --crosscheck   # also run GPT-5-mini subset

Idempotent/resumable: completed (model, condition, prompt_id, repeat_idx)
conversations already present in the output JSONL are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from . import conditions
from .conversation import run_conversation, turn_to_dict
from .judge import build_primary_judge, build_crosscheck_judge
from .models import build_target_model


_write_lock = threading.Lock()


def _output_path(model_key: str) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return os.path.join(config.RESULTS_DIR, f"responses_{model_key}.jsonl")


def _completed_conversations(path: str) -> set[tuple]:
    """Return the set of (condition, prompt_id, repeat_idx) already in the file."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["condition"], r["prompt_id"], r["repeat_idx"]))
    return done


def _append_results(path: str, rows: list[dict]) -> None:
    with _write_lock:
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def run_for_model(spec: config.ModelSpec, all_specs, judge, skip_existing: bool) -> int:
    path = _output_path(spec.key)
    done = _completed_conversations(path) if skip_existing else set()
    model = build_target_model(spec)

    todo = [
        s for s in all_specs
        if (s.condition, s.prompt_id, s.repeat_idx) not in done
    ]
    print(f"[{spec.key}] {len(todo)} conversations to run "
          f"({len(all_specs) - len(todo)} already done)")

    n_responses = 0

    def _work(cspec):
        results = run_conversation(cspec, model, judge)
        rows = [turn_to_dict(r) for r in results]
        _append_results(path, rows)
        return len(rows)

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futures = {ex.submit(_work, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                n_responses += fut.result()
            except Exception as e:  # noqa: BLE001
                cs = futures[fut]
                print(f"[{spec.key}] conversation failed "
                      f"({cs.condition}/{cs.prompt_id}#{cs.repeat_idx}): {e}")
            if i % 25 == 0:
                print(f"[{spec.key}] {i}/{len(todo)} conversations done")

    print(f"[{spec.key}] wrote {n_responses} responses -> {path}")
    return n_responses


def run_crosscheck(model_keys: list[str]) -> None:
    """Re-score a random sample of responses with the cross-check judge."""
    rng = random.Random(config.RANDOM_SEED)
    pool = []
    for key in model_keys:
        path = _output_path(key)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("frustration_score") is not None:
                    pool.append(r)
    if not pool:
        print("no scored responses found for cross-check")
        return

    rng.shuffle(pool)
    sample = pool[: config.CROSSCHECK_N]
    judge = build_crosscheck_judge()
    out_path = os.path.join(config.RESULTS_DIR, "crosscheck.jsonl")
    print(f"cross-checking {len(sample)} responses with {config.CROSSCHECK_MODEL_ID}")

    def _work(r):
        verdict = judge.score(r["assistant_response"])
        return {
            "model": r["model"],
            "condition": r["condition"],
            "prompt_id": r["prompt_id"],
            "repeat_idx": r["repeat_idx"],
            "turn_idx": r["turn_idx"],
            "primary_score": r["frustration_score"],
            "crosscheck_score": verdict["score"],
        }

    rows = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(_work, r) for r in sample]):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"crosscheck item failed: {e}")
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} cross-check rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Distress elicitation replication (Gemma + Gemini).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model keys to run (default: all in-scope)")
    parser.add_argument("--crosscheck", action="store_true",
                        help="also re-score a random subset with the cross-check judge")
    parser.add_argument("--no-skip", action="store_true",
                        help="do not skip already-completed conversations")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sample budget and exit without API calls")
    args = parser.parse_args()

    budget = conditions.response_budget(config.EVAL_SCALE)
    print("Sample budget (per model) at EVAL_SCALE=%.3f:" % config.EVAL_SCALE)
    for name, info in budget.items():
        if name.startswith("_"):
            continue
        print(f"  {name:18s} {info['conversations']:5d} convos x {info['turns']} turns "
              f"= {info['responses']:5d} responses")
    print(f"  TOTAL responses/model: {budget['_total_responses']}")

    if args.dry_run:
        return

    all_specs = conditions.build_all_specs(config.EVAL_SCALE, config.RANDOM_SEED)
    selected = config.TARGET_MODELS
    if args.models:
        selected = [m for m in config.TARGET_MODELS if m.key in set(args.models)]
        if not selected:
            raise SystemExit(f"no matching models in {[m.key for m in config.TARGET_MODELS]}")

    judge = build_primary_judge()
    for spec in selected:
        run_for_model(spec, all_specs, judge, skip_existing=not args.no_skip)

    if args.crosscheck:
        run_crosscheck([m.key for m in selected])


if __name__ == "__main__":
    main()
