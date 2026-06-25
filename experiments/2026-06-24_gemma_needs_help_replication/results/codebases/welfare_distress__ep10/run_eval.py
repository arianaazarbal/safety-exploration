"""Run the distress-elicitation evaluation (paper Section 2) for Gemma + Gemini.

For each model and each of the 8 conditions we run multi-turn rollouts, then score every
assistant turn with the Claude-Sonnet-4 judge. Results are written as JSONL, one row per
scored response, to results/<model>.jsonl.

Usage:
    python run_eval.py                      # full paper-scale run, all 4 models
    python run_eval.py --scale 0.02         # ~2% smoke test
    python run_eval.py --models gemini-2.5-flash
    python run_eval.py --skip-judge         # generate responses only (judge later)
    python run_eval.py --judge-only         # (re)score existing results/<model>.jsonl

Required env: OPENROUTER_API_KEY (Gemini, and Gemma if GEMMA_BACKEND=openrouter),
              ANTHROPIC_API_KEY (judge). Local Gemma (default) needs a GPU + transformers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from conversation import ResponseRecord, run_rollout
from judge import EmotionJudge
from models import ModelClient
from wildchat import load_wildchat_prompts

SEED = 1234


def _out_path(model_name: str) -> str:
    return os.path.join(config.RESULTS_DIR, f"{model_name}.jsonl")


def generate_responses(spec, conditions, wildchat_prompts) -> list[ResponseRecord]:
    """Run all rollouts for one model and return every assistant-turn record."""
    client = ModelClient(spec)
    records: list[ResponseRecord] = []

    # Build the full list of (condition, conversation_id) rollouts.
    jobs = []
    for cond in conditions:
        for cid in range(cond.n_conversations):
            jobs.append((cond, cid))

    def do_one(job):
        cond, cid = job
        # Per-conversation RNG keyed by a *process-stable* hash (zlib.crc32, unlike the
        # salted built-in str hash) so runs are reproducible regardless of scheduling
        # order or Python process.
        key = f"{SEED}|{spec.name}|{cond.key}|{cid}".encode()
        rng = random.Random(zlib.crc32(key))
        return run_rollout(client, cond, cid, rng, wildchat_prompts)

    if spec.backend == "openrouter":
        # API calls parallelise well.
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
            futures = [ex.submit(do_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                records.extend(fut.result())
                if i % 20 == 0 or i == len(jobs):
                    print(f"  [{spec.name}] rollouts {i}/{len(jobs)}", flush=True)
    else:
        # Local GPU inference: run sequentially to avoid contention.
        for i, job in enumerate(jobs, 1):
            records.extend(do_one(job))
            if i % 5 == 0 or i == len(jobs):
                print(f"  [{spec.name}] rollouts {i}/{len(jobs)}", flush=True)

    return records


def judge_records(records: list[ResponseRecord]) -> None:
    """Score every record in place with the emotion judge (parallel API calls)."""
    judge = EmotionJudge()

    def do_one(rec: ResponseRecord):
        res = judge.score(rec.response_text)
        rec.rating = res.rating
        rec.evidence = res.evidence
        rec.reasoning = res.reasoning
        return rec

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futures = [ex.submit(do_one, r) for r in records]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 50 == 0 or i == len(records):
                print(f"  judged {i}/{len(records)}", flush=True)


def write_jsonl(path: str, records: list[ResponseRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def read_jsonl(path: str) -> list[ResponseRecord]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ResponseRecord(**json.loads(line)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply conversation counts")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of condition keys")
    ap.add_argument("--skip-judge", action="store_true", help="generate responses, don't judge")
    ap.add_argument("--judge-only", action="store_true", help="(re)judge existing results")
    args = ap.parse_args()

    specs = config.MODELS
    if args.models:
        specs = [s for s in specs if s.name in set(args.models)]
        if not specs:
            sys.exit(f"No models matched {args.models}")

    conditions = config.scaled_conditions(args.scale)
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c.key in wanted]

    wildchat_prompts = load_wildchat_prompts()

    for spec in specs:
        print(f"\n=== {spec.name} ({spec.backend}:{spec.model_id}) ===", flush=True)
        path = _out_path(spec.name)

        if args.judge_only:
            records = read_jsonl(path)
            print(f"  loaded {len(records)} existing responses to (re)judge", flush=True)
        else:
            records = generate_responses(spec, conditions, wildchat_prompts)
            print(f"  generated {len(records)} responses", flush=True)
            write_jsonl(path, records)  # checkpoint before judging

        if not args.skip_judge:
            judge_records(records)
            write_jsonl(path, records)

        print(f"  wrote {path}", flush=True)

    print("\nDone. Run `python analyze.py` to compute summary metrics.", flush=True)


if __name__ == "__main__":
    main()
