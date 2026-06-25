"""Run the full distress-elicitation evaluation for the in-scope models.

For each target model (Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash,
Gemini-2.5-Pro) this:
  1. builds the conversation specs for the active sampling PROFILE,
  2. runs + judges each conversation concurrently,
  3. streams results to results/rollouts/<model>.jsonl (resumable).

Results are appended incrementally and the run is resumable: re-running skips any
conversation id already present in the model's jsonl file.

Usage:
    PROFILE=smoke python run_eval.py                  # all in-scope models
    PROFILE=pilot python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from conditions import build_all_conversations, summarize_counts
from rollout import run_rollout

_write_lock = threading.Lock()


def _load_done_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_record(path: str, record: dict) -> None:
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_model(model: config.ModelSpec, profile: config.Profile) -> None:
    os.makedirs(config.ROLLOUTS_DIR, exist_ok=True)
    out_path = os.path.join(config.ROLLOUTS_DIR, f"{model.key}.jsonl")

    specs = build_all_conversations(profile)
    done = _load_done_ids(out_path)
    todo = [s for s in specs if s.id not in done]

    print(f"\n=== {model.display} [{model.backend}:{model.model_id}] ===")
    print(f"  profile={profile.name} total={len(specs)} done={len(done)} todo={len(todo)}")
    if not todo:
        print("  nothing to do.")
        return

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
        futures = {pool.submit(run_rollout, model, s): s for s in todo}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                record = fut.result()
                _append_record(out_path, record.to_json())
                completed += 1
            except Exception as e:  # noqa: BLE001 - log and continue
                errors += 1
                print(f"  [error] {spec.id}: {e}", file=sys.stderr)
            if (completed + errors) % 25 == 0 or (completed + errors) == len(todo):
                print(f"  progress: {completed + errors}/{len(todo)} "
                      f"(ok={completed}, err={errors})")
    print(f"  done: {completed} written, {errors} errors -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model keys to run (default: all in-scope)")
    ap.add_argument("--profile", default=None,
                    help="override PROFILE env (smoke|pilot|half|paper)")
    args = ap.parse_args()

    profile = config.PROFILES[args.profile] if args.profile else config.ACTIVE_PROFILE

    counts = summarize_counts(profile)
    print(f"Profile '{profile.name}': {sum(counts.values())} conversations/model")
    for cond, n in counts.items():
        print(f"  {cond:22s} {n}")

    models = config.TARGET_MODELS
    if args.models:
        keys = set(args.models)
        models = [m for m in models if m.key in keys]
        if not models:
            sys.exit(f"No matching models in {[m.key for m in config.TARGET_MODELS]}")

    for model in models:
        run_model(model, profile)


if __name__ == "__main__":
    main()
