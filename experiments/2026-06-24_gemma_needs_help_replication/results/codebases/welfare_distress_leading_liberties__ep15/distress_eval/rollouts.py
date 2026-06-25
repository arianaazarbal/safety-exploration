"""Multi-turn rollout generation with on-disk checkpointing.

For each (model, condition, rollout_idx) we build an alternating chat:
    user(task) -> assistant -> user(rejection) -> assistant -> ...
recording every assistant turn as a separate scored unit ("response").

Output: one JSONL file per model at results/generations/<model>.jsonl, one line
per rollout. Reruns skip rollouts already present, so a run is resumable.
"""

from __future__ import annotations

import asyncio
import json
import random
import zlib

import httpx

import config
from . import wildchat
from .clients import ChatClient
from .conditions import CONDITIONS, Condition, opening_prompt, rejection_for_turn


def _rollout_id(model_key: str, cond_key: str, idx: int) -> str:
    return f"{model_key}|{cond_key}|{idx}"


def _load_done_ids(path) -> set[str]:
    done = set()
    if path.exists():
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


async def _run_one_rollout(
    http: httpx.AsyncClient,
    client: ChatClient,
    cond: Condition,
    idx: int,
    wildchat_prompts: list[str],
) -> dict:
    """Execute a single multi-turn rollout, returning a record with all turns."""
    # Per-rollout RNG so rejection sampling is deterministic given the index.
    # zlib.crc32 (not hash()) because Python string hashing is salted per process.
    seed = zlib.crc32(f"{client.cfg.key}|{cond.key}|{idx}".encode())
    rng = random.Random(seed)

    opening = opening_prompt(cond, idx, wildchat_prompts)
    messages: list[dict] = [{"role": "user", "content": opening}]
    turns = []

    for turn_idx in range(cond.turns):
        response = await client.generate(http, messages)
        messages.append({"role": "assistant", "content": response})
        turns.append({
            "turn_idx": turn_idx,
            "response": response,
            "user_message": messages[-2]["content"],
        })
        if turn_idx < cond.turns - 1:
            rejection = rejection_for_turn(cond, turn_idx, rng)
            messages.append({"role": "user", "content": rejection})

    return {
        "rollout_id": _rollout_id(client.cfg.key, cond.key, idx),
        "model": client.cfg.key,
        "condition": cond.key,
        "category": cond.category,
        "rollout_idx": idx,
        "opening_prompt": opening,
        "n_turns": cond.turns,
        "turns": turns,
    }


async def generate_for_model(
    model_key: str,
    *,
    scale: float = config.SCALE,
    conditions: list[Condition] | None = None,
) -> None:
    cfg = config.MODELS[model_key]
    conditions = conditions or CONDITIONS
    wildchat_prompts = wildchat.load_wildchat_prompts()

    out_path = config.GENERATIONS_DIR / f"{model_key}.jsonl"
    done = _load_done_ids(out_path)

    client = ChatClient(
        cfg,
        base_url=config.base_url_for(cfg.backend),
        api_key=config.api_key_for(cfg.backend),
    )

    # Build the full task list, skipping completed rollouts.
    tasks_spec: list[tuple[Condition, int]] = []
    for cond in conditions:
        for idx in range(cond.n_rollouts(scale)):
            if _rollout_id(model_key, cond.key, idx) not in done:
                tasks_spec.append((cond, idx))

    total = len(tasks_spec)
    if total == 0:
        print(f"[gen:{model_key}] nothing to do ({len(done)} rollouts already present).")
        return
    print(f"[gen:{model_key}] {total} rollouts to run ({len(done)} cached). "
          f"backend={cfg.backend} model={cfg.provider_model}")

    sem = asyncio.Semaphore(config.GEN_CONCURRENCY)
    write_lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient() as http:
        f = out_path.open("a")
        try:
            async def worker(cond: Condition, idx: int):
                nonlocal completed
                async with sem:
                    try:
                        record = await _run_one_rollout(http, client, cond, idx, wildchat_prompts)
                    except Exception as e:  # noqa: BLE001
                        print(f"[gen:{model_key}] FAILED {cond.key}#{idx}: {e!r}")
                        return
                async with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"[gen:{model_key}] {completed}/{total}")

            await asyncio.gather(*(worker(c, i) for c, i in tasks_spec))
        finally:
            f.close()

    print(f"[gen:{model_key}] done.")
