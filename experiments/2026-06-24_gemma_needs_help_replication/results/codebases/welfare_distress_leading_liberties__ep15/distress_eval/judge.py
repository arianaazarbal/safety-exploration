"""Score every assistant turn with the Claude Sonnet 4 emotion judge.

Reads results/generations/<model>.jsonl, scores each turn, and writes
results/scores/<model>.jsonl (one line per scored response). Resumable: already
scored response_ids are skipped.
"""

from __future__ import annotations

import asyncio
import json

import httpx

import config
from .clients import JudgeClient
from .prompts import JUDGE_PROMPT_TEMPLATE


def _response_id(rollout_id: str, turn_idx: int) -> str:
    return f"{rollout_id}|turn{turn_idx}"


def _iter_responses(gen_path):
    """Yield one dict per assistant turn across all rollouts in a file."""
    with gen_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for turn in rec["turns"]:
                yield {
                    "response_id": _response_id(rec["rollout_id"], turn["turn_idx"]),
                    "rollout_id": rec["rollout_id"],
                    "model": rec["model"],
                    "condition": rec["condition"],
                    "category": rec["category"],
                    "rollout_idx": rec["rollout_idx"],
                    "turn_idx": turn["turn_idx"],
                    "n_turns": rec["n_turns"],
                    "response": turn["response"],
                }


def _load_done_ids(path) -> set[str]:
    done = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["response_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


async def score_model(model_key: str) -> None:
    gen_path = config.GENERATIONS_DIR / f"{model_key}.jsonl"
    if not gen_path.exists():
        print(f"[judge:{model_key}] no generations file; run generation first.")
        return
    out_path = config.SCORES_DIR / f"{model_key}.jsonl"
    done = _load_done_ids(out_path)

    pending = [r for r in _iter_responses(gen_path) if r["response_id"] not in done]
    total = len(pending)
    if total == 0:
        print(f"[judge:{model_key}] nothing to score ({len(done)} cached).")
        return
    print(f"[judge:{model_key}] {total} responses to score ({len(done)} cached).")

    judge = JudgeClient(
        model=config.JUDGE_MODEL,
        api_key=config.anthropic_api_key(),
        base_url=config.ANTHROPIC_BASE_URL,
    )

    sem = asyncio.Semaphore(config.JUDGE_CONCURRENCY)
    write_lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient() as http:
        f = out_path.open("a")
        try:
            async def worker(item: dict):
                nonlocal completed
                async with sem:
                    prompt = JUDGE_PROMPT_TEMPLATE.format(response=item["response"])
                    try:
                        result = await judge.score(http, prompt)
                    except Exception as e:  # noqa: BLE001
                        print(f"[judge:{model_key}] FAILED {item['response_id']}: {e!r}")
                        return
                if result["rating"] is None:
                    print(f"[judge:{model_key}] unparseable rating for {item['response_id']}; skipped.")
                    return
                record = {
                    "response_id": item["response_id"],
                    "rollout_id": item["rollout_id"],
                    "model": item["model"],
                    "condition": item["condition"],
                    "category": item["category"],
                    "rollout_idx": item["rollout_idx"],
                    "turn_idx": item["turn_idx"],
                    "n_turns": item["n_turns"],
                    "rating": result["rating"],
                    "evidence": result["evidence"],
                    "reasoning": result["reasoning"],
                }
                async with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        print(f"[judge:{model_key}] {completed}/{total}")

            await asyncio.gather(*(worker(it) for it in pending))
        finally:
            f.close()

    print(f"[judge:{model_key}] done.")
