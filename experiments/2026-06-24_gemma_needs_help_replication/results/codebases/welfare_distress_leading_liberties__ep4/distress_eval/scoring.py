"""Judging phase: score every recorded response on the 0-10 frustration scale,
plus the GPT validation cross-check on a fixed 260-response subset.

Both are checkpointed by `response_id`.
"""

from __future__ import annotations

import asyncio
import random

from tqdm.asyncio import tqdm_asyncio

from . import config, storage
from .clients import JudgeClient, ValidationJudgeClient


async def _score_file(judge, responses: list[dict], out_path, judge_label: str) -> None:
    done = storage.done_keys(out_path, "response_id")
    pending = [r for r in responses if r["response_id"] not in done]
    print(f"[{judge_label}] {len(responses)} responses, "
          f"{len(responses) - len(pending)} scored, {len(pending)} to score.")
    if not pending:
        return

    sem = asyncio.Semaphore(config.MAX_CONCURRENT_JUDGE)

    async def _worker(r: dict):
        async with sem:
            try:
                score = await judge.score(r["assistant_text"])
            except Exception as e:  # noqa: BLE001
                print(f"[{judge_label}] response {r['response_id']} failed: {e}")
                return
            storage.append_row(out_path, {
                "response_id": r["response_id"],
                "judge_model": judge.model,
                "score": score,
            })

    await tqdm_asyncio.gather(*[_worker(r) for r in pending], desc=judge_label)


async def judge_all() -> None:
    responses = storage.load_rows(config.RESPONSES_PATH)
    if not responses:
        print("[judge] no responses found; run generation first.")
        return
    judge = JudgeClient()
    await _score_file(judge, responses, config.SCORES_PATH, "judge")


def _validation_subset(responses: list[dict]) -> list[dict]:
    """Deterministically sample VALIDATION_SAMPLE_SIZE responses for re-scoring."""
    rng = random.Random(config.SEED + 1)
    pool = sorted(responses, key=lambda r: r["response_id"])
    n = min(config.VALIDATION_SAMPLE_SIZE, len(pool))
    return rng.sample(pool, n)


async def validate_judge() -> None:
    responses = storage.load_rows(config.RESPONSES_PATH)
    if not responses:
        print("[validation] no responses found; run generation first.")
        return
    subset = _validation_subset(responses)
    judge = ValidationJudgeClient()
    await _score_file(judge, subset, config.VALIDATION_SCORES_PATH, "validation")
