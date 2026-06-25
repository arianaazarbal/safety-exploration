"""Orchestrate the full distress-elicitation sweep for the in-scope models.

For each model we build all rollout specs (conditions.build_rollout_specs), run them with
bounded concurrency, judge every turn, and stream results to a JSONL file per model. The
run is resumable: rollouts whose ids already appear in the output file are skipped.
"""

from __future__ import annotations

import asyncio
import json
import os

from tqdm.asyncio import tqdm_asyncio

import config
from clients import JudgeClient, TargetClient
from conditions import build_rollout_specs
from conversation import run_rollout


def _raw_path(model_key: str) -> str:
    return os.path.join(config.RAW_DIR, f"{model_key}.jsonl")


def _completed_rollout_ids(path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # a rollout is "done" only once we've seen its final turn
                if rec.get("turn") == rec.get("n_turns"):
                    done.add(rec["rollout_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def evaluate_model(model_key: str, seed: int = config.SEED) -> str:
    """Run all conditions for one model. Returns the path to its JSONL results."""
    model = config.TARGET_MODELS_BY_KEY[model_key]
    os.makedirs(config.RAW_DIR, exist_ok=True)
    out_path = _raw_path(model_key)

    specs = build_rollout_specs(model_key, seed=seed)
    done = _completed_rollout_ids(out_path)
    pending = [s for s in specs if s.rollout_id not in done]

    print(f"[{model_key}] {len(specs)} rollouts total, {len(done)} already done, "
          f"{len(pending)} to run.")
    if not pending:
        return out_path

    target = TargetClient()
    judge = JudgeClient()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_ROLLOUTS)
    write_lock = asyncio.Lock()
    fh = open(out_path, "a", encoding="utf-8")

    async def worker(spec):
        async with sem:
            try:
                records = await run_rollout(spec, target, judge, model.provider_id)
            except Exception as exc:  # noqa: BLE001 - log and continue with other rollouts
                print(f"[{model_key}] rollout {spec.rollout_id} ({spec.condition}) "
                      f"failed: {exc}")
                return
            async with write_lock:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    try:
        await tqdm_asyncio.gather(*[worker(s) for s in pending],
                                  desc=f"{model_key}", unit="rollout")
    finally:
        fh.close()

    return out_path


async def evaluate_all(model_keys: list[str] | None = None, seed: int = config.SEED):
    keys = model_keys or [m.key for m in config.TARGET_MODELS]
    for key in keys:
        await evaluate_model(key, seed=seed)
