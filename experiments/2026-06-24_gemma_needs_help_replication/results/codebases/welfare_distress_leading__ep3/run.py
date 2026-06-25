#!/usr/bin/env python3
"""Generate and judge distress-elicitation rollouts for the in-scope models.

Usage examples
--------------
    # Cheap pilot over all four models (default scale)
    python run.py

    # One model, full paper scale
    python run.py --models gemma-3-27b-it --scale paper

    # Resume an interrupted run (skips rollouts already in the JSONL)
    python run.py --models gemini-2.5-flash --scale paper

Results are written to ``results/<model>__<scale>.jsonl`` — one JSON object per
rollout, including every judged assistant turn. Re-running appends only the
rollouts that are missing, so a run is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from distress import conditions, wildchat
from distress.backends import make_backend
from distress.judge import Judge
from distress.rollout import run_rollout


def _load_done_ids(path) -> set[int]:
    """Rollout ids already present (and error-free) in an existing JSONL."""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                continue  # let failed rollouts be retried
            rid = obj.get("meta", {}).get("rollout_id")
            if rid is not None:
                done.add(rid)
    return done


def run_model(model_key: str, scale: str, specs, judge: Judge) -> None:
    spec_model = config.MODELS[model_key]
    backend = make_backend(model_key)
    path = config.results_path(model_key, scale)

    done = _load_done_ids(path)
    todo = [s for s in specs if s.meta["rollout_id"] not in done]
    if not todo:
        print(f"[{model_key}] all {len(specs)} rollouts already complete -> {path}")
        return
    print(
        f"[{model_key}] {len(todo)} rollouts to run "
        f"({len(done)} already done) -> {path}"
    )

    write_lock = threading.Lock()
    fh = path.open("a")

    def work(spec):
        return run_rollout(
            spec,
            model_key,
            backend,
            judge,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            max_tokens=config.GEN_MAX_TOKENS,
            disable_thinking=spec_model.disable_thinking,
        )

    n_err = 0
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            futures = {pool.submit(work, s): s for s in todo}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=model_key, unit="rollout"
            ):
                res = fut.result()
                if res.error:
                    n_err += 1
                with write_lock:
                    fh.write(json.dumps(res.to_json(), ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        fh.close()
    if n_err:
        print(f"[{model_key}] completed with {n_err} errored rollouts (re-run to retry).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        choices=list(config.MODELS),
        help="Subset of in-scope models to run.",
    )
    p.add_argument(
        "--scale",
        default=config.DEFAULT_SCALE,
        choices=list(config.SCALE_PRESETS),
        help="Sampling scale preset.",
    )
    p.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="RNG seed for sampling."
    )
    args = p.parse_args(argv)

    counts = config.rollout_counts(args.scale)
    total = sum(counts.values())
    print(f"Scale '{args.scale}': {counts}  (= {total} rollouts/model)")

    # Build the shared condition set once (same across models for comparability).
    wc_prompts = wildchat.load_wildchat_prompts(
        n=config.WILDCHAT_N_PROMPTS, seed=args.seed, dataset_name=config.WILDCHAT_DATASET
    )
    print(f"WildChat: loaded {len(wc_prompts)} prompts.")
    specs = conditions.build_all(counts, seed=args.seed, wildchat_prompts=wc_prompts)

    judge = Judge(config.PRIMARY_JUDGE)
    print(f"Judge: {config.PRIMARY_JUDGE.name} ({config.PRIMARY_JUDGE.model_id})")

    for model_key in args.models:
        run_model(model_key, args.scale, specs, judge)

    print("\nDone. Analyze with:  python analyze.py --scale", args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
