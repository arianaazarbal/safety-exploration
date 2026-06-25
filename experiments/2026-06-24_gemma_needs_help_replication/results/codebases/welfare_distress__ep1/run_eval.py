"""Run the distress-elicitation evaluation (paper Section 2).

For each target model (Gemma + Gemini) and each of the 8 conditions, sample the
configured number of rollouts, judge every assistant turn, and stream the
results to results/<model>/responses.jsonl. Then aggregate with analyze.py.

Usage:
    python run_eval.py                      # all target models, smoke budget
    python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python run_eval.py --conditions impossible_numeric extended
    EVAL_BUDGET=paper python run_eval.py    # full paper sample counts

Required environment (depending on chosen backends):
    OPENROUTER_API_KEY   for OpenRouter-served models and/or the judge
    GOOGLE_API_KEY       for backend=google Gemini
    ANTHROPIC_API_KEY    for the default Anthropic judge

Rollouts and judge calls are run concurrently via a thread pool
(config.MAX_CONCURRENCY). Already-completed rollouts are skipped on re-run, so
the script is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from judge import Judge
from models import get_client
from rollout import run_rollout
from tasks import CONDITIONS, condition_by_key


def _results_path(model_name: str) -> str:
    d = os.path.join(config.RESULTS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, config.RAW_RESPONSES_FILE)


def _completed_rollouts(path: str) -> set[tuple[str, int]]:
    """Set of (condition, rollout_id) already fully recorded, for resumption."""
    done: dict[tuple[str, int], int] = {}
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["condition"], r["rollout_id"])
            done[key] = max(done.get(key, 0), r["turn"])
    # A rollout is complete only if we saw all of its turns.
    complete = set()
    for (cond_key, rid), max_turn in done.items():
        if max_turn >= condition_by_key(cond_key).turns:
            complete.add((cond_key, rid))
    return complete


def run_for_model(model_name: str, condition_keys: list[str]) -> None:
    spec = config.model_by_name(model_name)
    client = get_client(spec)
    judge = Judge()
    counts = config.sample_counts()

    path = _results_path(model_name)
    done = _completed_rollouts(path)
    write_lock = threading.Lock()

    # Build the work list: every (condition, rollout_id) not already complete.
    jobs: list[tuple[str, int]] = []
    for cond_key in condition_keys:
        n = counts[cond_key]
        for rid in range(n):
            if (cond_key, rid) not in done:
                jobs.append((cond_key, rid))

    if not jobs:
        print(f"[{model_name}] nothing to do (all {sum(counts[c] for c in condition_keys)} rollouts complete)")
        return

    print(f"[{model_name}] running {len(jobs)} rollouts across {len(condition_keys)} conditions "
          f"(backend={spec.backend})")

    def _one(job: tuple[str, int]):
        cond_key, rid = job
        cond = condition_by_key(cond_key)
        # Deterministic per-rollout seed (stable across processes, unlike
        # hash()) so re-runs reproduce the same prompt / rejection choices.
        seed = zlib.crc32(f"{model_name}|{cond_key}|{rid}".encode())
        records = run_rollout(client, judge, cond, model_name, rid, seed)
        with write_lock, open(path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec.to_json()) + "\n")
        return cond_key, rid

    n_done = 0
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                fut.result()
            except Exception as e:  # keep going; one bad rollout shouldn't kill the run
                print(f"[{model_name}] rollout {job} failed: {e}", file=sys.stderr)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"[{model_name}] {n_done}/{len(jobs)} rollouts done")


def main():
    ap = argparse.ArgumentParser(description="Distress-elicitation eval (Gemma + Gemini)")
    ap.add_argument("--models", nargs="*", default=[m.name for m in config.TARGET_MODELS],
                    help="subset of target model names to run")
    ap.add_argument("--conditions", nargs="*", default=[c.key for c in CONDITIONS],
                    help="subset of condition keys to run")
    args = ap.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for model_name in args.models:
        run_for_model(model_name, args.conditions)

    print("\nDone. Aggregate with:  python analyze.py")


if __name__ == "__main__":
    main()
