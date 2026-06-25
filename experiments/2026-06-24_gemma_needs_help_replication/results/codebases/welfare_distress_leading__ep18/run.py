"""Orchestrate distress elicitation + judging for Gemma and Gemini models.

Pipeline per work unit (one rollout = one model x condition x rollout index):
  1. generate all assistant turns (multi-turn rejection protocol),
  2. score each turn with the Claude-Sonnet-4 judge,
  3. append one JSONL row per scored turn to results/responses.jsonl.

The run is checkpointed and resumable: completed rollouts are detected from the
existing JSONL and skipped. Use --scale for cheap smoke tests before the full
~4000-responses-per-model sweep.

Usage:
  export OPENROUTER_API_KEY=...
  python run.py --scale 0.02                 # quick smoke test
  python run.py                              # full paper scale
  python run.py --models gemma-3-27b-it --conditions extended numeric
  python analyze.py                          # tables after a run
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import time
from dataclasses import asdict

import config
import conversation
import judge
import models
from tqdm import tqdm


def _out_path() -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return os.path.join(config.OUTPUT_DIR, config.RESPONSES_FILE)


def _completed_rollouts(path: str) -> set[tuple[str, str, int]]:
    """Set of (model_key, cond_key, rollout_idx) fully present in the JSONL.

    A rollout counts as complete when it has rows for all of its turns.
    """
    if not os.path.exists(path):
        return set()
    counts: dict[tuple[str, str, int], set[int]] = {}
    expected: dict[tuple[str, str, int], int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = (row["model"], row["cond_key"], row["rollout_idx"])
            counts.setdefault(key, set()).add(row["turn"])
            expected[key] = row.get("n_turns", row["turn"])
    return {k for k, turns in counts.items() if len(turns) >= expected.get(k, 10**9)}


def _build_jobs(model_keys, cond_keys, scale):
    models_ = [m for m in config.TARGET_MODELS if m.key in model_keys]
    conds = [c for c in config.CONDITIONS if c.key in cond_keys]
    jobs = []
    for m in models_:
        for c in conds:
            n = config.n_rollouts(c, scale)
            for idx in range(n):
                jobs.append((m, c, idx))
    return jobs


async def _process_rollout(client, model_spec, cond, idx) -> list[dict]:
    """Generate + judge one rollout; return list of JSONL row dicts."""
    plan = conversation.build_rollout(cond, idx)

    chat_fn = functools.partial(
        models.chat,
        client,
        model=model_spec.or_id,
        temperature=config.TARGET_TEMPERATURE,
        max_tokens=config.TARGET_MAX_TOKENS,
        disable_thinking=model_spec.disable_thinking,
    )

    async def _chat(messages):
        return await chat_fn(messages=messages)

    turns = await conversation.run_rollout(_chat, plan)

    # Judge each turn (concurrently within the rollout; global semaphore still caps).
    scored = await asyncio.gather(*[judge.score_response(client, tr.response) for tr in turns])

    rows = []
    now = time.time()
    for tr, sc in zip(turns, scored):
        preceding = plan.rejections[tr.turn - 2] if tr.turn >= 2 else None
        rows.append(
            {
                "model": model_spec.key,
                "model_id": model_spec.or_id,
                "family": model_spec.family,
                "category": plan.category,
                "cond_key": plan.cond_key,
                "rollout_idx": plan.rollout_idx,
                "task_key": plan.task_key,
                "turn": tr.turn,
                "n_turns": plan.turns,
                "preceding_rejection": preceding,
                "first_user": plan.first_user if tr.turn == 1 else None,
                "response": tr.response,
                "rating": sc["rating"],
                "judge_evidence": sc["evidence"],
                "judge_reasoning": sc["reasoning"],
                "judge_error": sc["error"],
                "ts": now,
            }
        )
    return rows


async def _run(args):
    client = models.make_client()
    path = _out_path()

    done = set() if args.no_resume else _completed_rollouts(path)
    jobs = _build_jobs(args.models, args.conditions, args.scale)
    pending = [j for j in jobs if (j[0].key, j[1].key, j[2]) not in done]

    total_turns = sum(c.turns for (_, c, _) in pending)
    print(
        f"Models: {args.models}\nConditions: {args.conditions}\nScale: {args.scale}\n"
        f"Rollouts (=responses): {len(jobs)} total, {len(done)} already complete, "
        f"{len(pending)} to run\n"
        f"~{total_turns} assistant turns to generate, each judged "
        f"(~{total_turns} generation + ~{total_turns} judge calls)\nOutput: {path}\n"
    )
    if args.dry_run:
        print("Dry run; exiting before any API calls.")
        return
    if not pending:
        print("Nothing to do.")
        return

    write_lock = asyncio.Lock()
    fh = open(path, "a")
    pbar = tqdm(total=len(pending), desc="rollouts")

    async def worker(job):
        m, c, idx = job
        try:
            rows = await _process_rollout(client, m, c, idx)
        except Exception as exc:  # noqa: BLE001
            pbar.write(f"[error] {m.key}/{c.key}#{idx}: {type(exc).__name__}: {exc}")
            pbar.update(1)
            return
        async with write_lock:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        pbar.update(1)

    # Outer task fan-out; the models semaphore caps real API concurrency.
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY * 2)

    async def guarded(job):
        async with sem:
            await worker(job)

    try:
        await asyncio.gather(*[guarded(j) for j in pending])
    finally:
        pbar.close()
        fh.close()
        await client.close()

    print(f"\nDone. Run `python analyze.py` to compute tables.")


def parse_args():
    p = argparse.ArgumentParser(description="Distress elicitation for Gemma/Gemini.")
    p.add_argument("--scale", type=float, default=config.SCALE,
                   help="Fraction of paper per-condition response budget (1.0=full).")
    p.add_argument("--models", nargs="+", default=[m.key for m in config.TARGET_MODELS],
                   choices=[m.key for m in config.TARGET_MODELS])
    p.add_argument("--conditions", nargs="+", default=[c.key for c in config.CONDITIONS],
                   choices=[c.key for c in config.CONDITIONS])
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
