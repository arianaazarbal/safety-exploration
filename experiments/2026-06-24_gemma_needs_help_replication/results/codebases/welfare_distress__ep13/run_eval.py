"""Orchestrator: elicit and score distress across Gemma/Gemini models.

For every (model, condition, rollout) it builds the inputs, runs the multi-turn
rollout, scores each assistant turn with the judge, and streams results to disk.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python run_eval.py --profile smoke
    python run_eval.py --profile paper --models google/gemma-3-27b-it
    python analyze.py results/responses.jsonl

Results are written incrementally to:
    <output_dir>/responses.jsonl      one JSON line per scored assistant turn
    <output_dir>/conversations.jsonl  one JSON line per full rollout transcript
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from typing import List, TextIO

from client import LLMClient
from conditions import Condition, build_conditions
from config import Config, PROFILES, TARGET_MODELS
from rollout import RolloutRecord, run_rollout


def _rollout_rng(cfg: Config, model: str, condition_id: str, rollout_id: int) -> random.Random:
    """Deterministic per-rollout RNG so a run is fully reproducible."""
    key = f"{cfg.seed}|{model}|{condition_id}|{rollout_id}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _write_response_lines(fh: TextIO, rec: RolloutRecord) -> None:
    for t in rec.turns:
        line = {
            "model": rec.model,
            "category": rec.category,
            "condition": rec.condition,
            "rollout_id": rec.rollout_id,
            "prompt_id": rec.prompt_id,
            "turn": t.turn,
            "n_turns": len(rec.turns),
            "score": t.score,
            "evidence": t.evidence,
            "response": t.response,
        }
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _write_conversation_line(fh: TextIO, rec: RolloutRecord) -> None:
    line = {
        "model": rec.model,
        "category": rec.category,
        "condition": rec.condition,
        "rollout_id": rec.rollout_id,
        "prompt_id": rec.prompt_id,
        "task_prompt": rec.task_prompt,
        "rejections": rec.rejections,
        "messages": rec.messages,
        "scores": [t.score for t in rec.turns],
    }
    fh.write(json.dumps(line, ensure_ascii=False) + "\n")


async def main_async(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    responses_path = os.path.join(cfg.output_dir, cfg.results_filename)
    conversations_path = os.path.join(cfg.output_dir, cfg.conversations_filename)

    conditions = build_conditions(cfg.seed)
    profile = cfg.profile

    # Build the full work list up front so we can report totals/progress.
    work: List[tuple] = []  # (model, Condition, rollout_id)
    for model in cfg.target_models:
        for cond_id, count in profile.rollouts_per_condition.items():
            cond = conditions[cond_id]
            for rid in range(count):
                work.append((model, cond, rid))

    total = len(work)
    print(f"Profile '{profile.name}': {total} rollouts across "
          f"{len(cfg.target_models)} models and {len(conditions)} conditions.")

    client = LLMClient(cfg)
    write_lock = asyncio.Lock()
    # Bound the number of rollouts in flight; the client semaphore separately
    # bounds raw API concurrency (each rollout fans out judge calls).
    rollout_sem = asyncio.Semaphore(max(1, cfg.concurrency))
    done = 0

    with open(responses_path, "w", encoding="utf-8") as rf, \
         open(conversations_path, "w", encoding="utf-8") as cf:

        async def do_one(model: str, cond: Condition, rid: int) -> None:
            nonlocal done
            async with rollout_sem:
                rng = _rollout_rng(cfg, model, cond.id, rid)
                prompt_id, task_prompt, rejections = cond.build(rng)
                rec = await run_rollout(
                    client, cfg, model, cond, rid, prompt_id, task_prompt, rejections,
                )
            async with write_lock:
                _write_response_lines(rf, rec)
                _write_conversation_line(cf, rec)
                rf.flush()
                cf.flush()
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  ... {done}/{total} rollouts complete")

        await asyncio.gather(*(do_one(m, c, r) for (m, c, r) in work))

    await client.aclose()
    print(f"Done. Responses -> {responses_path}")
    print(f"      Transcripts -> {conversations_path}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=list(PROFILES), default="smoke",
                   help="sampling scale (default: smoke)")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"subset of target models (default: all of {TARGET_MODELS})")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    cfg = Config()
    cfg.profile_name = args.profile
    if args.models:
        cfg.target_models = args.models
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_tokens:
        cfg.max_tokens = args.max_tokens
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(main_async(cfg))
