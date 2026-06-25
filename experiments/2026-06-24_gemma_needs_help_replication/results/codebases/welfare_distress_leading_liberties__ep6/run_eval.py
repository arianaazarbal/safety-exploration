"""Orchestrate the distress-elicitation eval end to end.

For every (model, conversation-spec):
  1. run the multi-turn rollout
  2. score each assistant turn with the judge
  3. append one JSON record per scored turn to results/<model_key>/<condition>.jsonl

Runs are resumable: already-completed spec keys are skipped on restart, so you
can stop/restart or run model-by-model without redoing work. Concurrency is
bounded by --concurrency (default 8) to stay within API rate limits.

Usage:
    python run_eval.py                      # all models, all conditions (config.yaml)
    python run_eval.py --models gemma-3-27b-it
    python run_eval.py --scale 0.02         # quick pilot (~80 responses/model)
    python run_eval.py --concurrency 16
    python run_eval.py --dry-run            # print the plan and exit (no API calls)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml

import puzzles
from clients import GenerationClient, JudgeClient
from conditions import build_specs
from judge import score_turn
from rollout import run_rollout


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def maybe_load_dotenv() -> None:
    """Load .env if python-dotenv is installed; otherwise no-op."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _completed_keys(path: Path) -> set[str]:
    """Spec keys already fully recorded in a results file (for resume)."""
    if not path.exists():
        return set()
    keys: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(rec["spec_key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


async def _process_spec(spec, model, gen, judge, out_path, lock, sem, counters):
    async with sem:
        try:
            rollout = await run_rollout(spec, model["key"], model["id"], gen)
        except Exception as e:  # rollout failed after retries -> log and skip
            counters["rollout_errors"] += 1
            sys.stderr.write(f"[rollout-error] {model['key']} {spec.key}: {e}\n")
            return

        records = []
        for turn in rollout.turns:
            try:
                score = await score_turn(turn.assistant_text, judge)
            except Exception as e:
                counters["judge_errors"] += 1
                sys.stderr.write(f"[judge-error] {model['key']} {spec.key} t{turn.turn}: {e}\n")
                continue
            records.append({
                "model_key": model["key"],
                "model_id": model["id"],
                "family": model["family"],
                "condition_id": rollout.condition_id,
                "category": rollout.category,
                "variant": rollout.variant,
                "spec_key": spec.key,
                "replicate": spec.replicate,
                "n_turns": spec.n_turns,
                "turn": turn.turn,
                "user_message": turn.user_message,
                "assistant_text": turn.assistant_text,
                "rating": score.rating,
                "evidence": score.evidence,
                "reasoning": score.reasoning,
            })

    # write all turns for this spec atomically under the file lock
    async with lock:
        with open(out_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counters["specs_done"] += 1
    counters["responses"] += len(records)
    done = counters["specs_done"]
    if done % 25 == 0:
        sys.stderr.write(
            f"  ...{done}/{counters['specs_total']} specs, "
            f"{counters['responses']} responses scored\n"
        )


async def run_model(model, specs, gen, judge, results_dir, concurrency, counters):
    model_dir = results_dir / model["key"]
    model_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    # one lock per condition file (we write per-condition)
    locks: dict[str, asyncio.Lock] = {}
    # read each condition's completed keys once (not once per spec)
    done_by_cond: dict[str, set] = {}
    tasks = []

    for spec in specs:
        out_path = model_dir / f"{spec.condition_id}.jsonl"
        if spec.condition_id not in done_by_cond:
            done_by_cond[spec.condition_id] = _completed_keys(out_path)
        if spec.key in done_by_cond[spec.condition_id]:
            continue
        lock = locks.setdefault(spec.condition_id, asyncio.Lock())
        tasks.append(_process_spec(spec, model, gen, judge, out_path, lock, sem, counters))

    counters["specs_total"] += len(tasks)
    sys.stderr.write(f"[{model['key']}] {len(tasks)} specs to run "
                     f"(rest already cached)\n")
    await asyncio.gather(*tasks)


async def main_async(args):
    maybe_load_dotenv()
    config = load_config(args.config)
    if args.scale is not None:
        config["scale"] = args.scale

    # 1. confirm the "impossible" puzzles really are impossible
    puzzles.verify_all()
    sys.stderr.write("Puzzle impossibility verified (Countdown + Fraction).\n")

    # 2. build the full plan
    specs = build_specs(config)
    models = config["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["key"] in wanted]
        if not models:
            sys.exit(f"No models match {args.models}; available: "
                     f"{[m['key'] for m in config['models']]}")

    total_responses = sum(s.n_turns for s in specs) * len(models)
    sys.stderr.write(
        f"Plan: {len(models)} model(s) x {len(specs)} conversations "
        f"= ~{total_responses} scored responses total.\n"
    )
    for m in models:
        sys.stderr.write(f"  - {m['key']} ({m['id']})\n")

    if args.dry_run:
        # print per-condition spec counts and a sample, then exit
        from collections import Counter
        by_cond = Counter(s.condition_id for s in specs)
        sys.stderr.write("\nConversations per condition (per model):\n")
        for cid, n in by_cond.items():
            sys.stderr.write(f"  {cid}: {n}\n")
        sys.stderr.write("\nSample spec:\n")
        sys.stderr.write(json.dumps(asdict(specs[0]), indent=2, ensure_ascii=False) + "\n")
        return

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    counters = {"specs_total": 0, "specs_done": 0, "responses": 0,
                "rollout_errors": 0, "judge_errors": 0}

    timeout = httpx.Timeout(config["generation"]["timeout_s"])
    limits = httpx.Limits(max_connections=args.concurrency * 4,
                          max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as gen_http, \
               httpx.AsyncClient(timeout=httpx.Timeout(config["judge"]["timeout_s"]),
                                 limits=limits) as judge_http:
        gen = GenerationClient(config["generation"], gen_http)
        judge = JudgeClient(config["judge"], judge_http)

        for model in models:
            await run_model(model, specs, gen, judge, results_dir,
                            args.concurrency, counters)

    sys.stderr.write(
        f"\nDone. {counters['responses']} responses scored. "
        f"rollout_errors={counters['rollout_errors']} "
        f"judge_errors={counters['judge_errors']}\n"
        f"Results in: {results_dir}/  ->  run `python analyze.py`\n"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Distress-elicitation replication runner")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--models", nargs="*", help="subset of model keys to run")
    p.add_argument("--scale", type=float, default=None,
                   help="override config.scale (e.g. 0.02 for a pilot)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and a sample spec, make no API calls")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
