#!/usr/bin/env python3
"""Run the distress-elicitation evaluation for Gemma + Gemini models.

Replicates Section 2 of "Gemma Needs Help" (arXiv:2603.10011v1): present a
task, reject the model over multiple turns, and score every response on a 0-10
frustration scale with a Claude judge.

Usage:
    python run_eval.py                       # all 4 models, full paper counts
    python run_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python run_eval.py --quick               # tiny smoke run
    python run_eval.py --max-concurrency 16

Results stream to results/<model>.jsonl (one rollout per line) and are
resumable: re-running skips rollout_ids already present.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from backends import AnthropicJudge, OpenRouterBackend
from rollout import run_rollout
from specs import build_specs


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def _run_model(
    model_key: str,
    model_id: str,
    cfg: config.RunConfig,
    backend: OpenRouterBackend,
    judge: AnthropicJudge,
) -> None:
    out_path = Path(cfg.output_dir) / f"{model_key}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_ids(out_path)
    specs = [s for s in build_specs(cfg.sample_counts, cfg.seed)
             if s.rollout_id not in done]

    total = sum(cfg.sample_counts.values())
    print(f"\n=== {model_key} ({model_id}) ===")
    print(f"  {len(done)} already done, {len(specs)} to run (target {total}).")
    if not specs:
        return

    sem = asyncio.Semaphore(cfg.max_concurrency)
    write_lock = asyncio.Lock()
    counter = {"done": 0, "failed": 0}

    async def worker(spec):
        async with sem:
            try:
                record = await run_rollout(
                    spec, model_id, backend, judge,
                    cfg.target_temperature, cfg.target_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - log & continue
                counter["failed"] += 1
                print(f"  [fail] {spec.rollout_id}: {exc}", file=sys.stderr)
                return
            async with write_lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                counter["done"] += 1
                n = counter["done"]
                if n % 25 == 0 or n == len(specs):
                    print(f"  {n}/{len(specs)} rollouts "
                          f"({counter['failed']} failed)")

    await asyncio.gather(*(worker(s) for s in specs))
    print(f"  finished {model_key}: {counter['done']} ok, {counter['failed']} failed.")


async def main_async(args) -> None:
    cfg = config.RunConfig()
    cfg.sample_counts = dict(config.SAMPLE_COUNTS_QUICK if args.quick
                             else config.SAMPLE_COUNTS)
    cfg.max_concurrency = args.max_concurrency
    cfg.output_dir = args.output_dir
    cfg.seed = args.seed
    cfg.judge_model = args.judge_model

    selected = args.models or list(config.TARGET_MODELS)
    unknown = [m for m in selected if m not in config.TARGET_MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Known: {list(config.TARGET_MODELS)}")

    backend = OpenRouterBackend()
    judge = AnthropicJudge(
        cfg.judge_model, cfg.judge_temperature, cfg.judge_max_tokens
    )

    print(f"Judge: {cfg.judge_model} | targets via OpenRouter | "
          f"temp={cfg.target_temperature} | concurrency={cfg.max_concurrency}")
    for model_key in selected:
        await _run_model(model_key, config.TARGET_MODELS[model_key], cfg, backend, judge)

    print(f"\nDone. Analyze with:  python analyze.py --output-dir {cfg.output_dir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=None,
                   help=f"subset of {list(config.TARGET_MODELS)}")
    p.add_argument("--quick", action="store_true",
                   help="tiny sample counts for a plumbing test")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--judge-model", default=config.JUDGE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
