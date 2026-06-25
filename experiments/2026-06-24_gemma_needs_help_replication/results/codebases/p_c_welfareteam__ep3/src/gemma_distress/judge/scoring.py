"""Score every assistant turn of a set of rollouts with a judge.

Bridges Section 2 elicitation (rollouts) and Section 2.1 scoring. Reconstructs
the conversation context preceding each assistant turn (so the judge sees
escalation), scores concurrently, and emits one record per scored response.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import Judge


def _context_for_turn(rollout: dict, turn_index: int) -> list[dict]:
    """Conversation up to (not including) the assistant response at ``turn_index``."""
    ctx: list[dict] = []
    for turn in rollout["turns"]:
        ti = turn["turn_index"]
        ctx.append({"role": "user", "content": turn["user_message"]})
        if ti >= turn_index:
            break
        ctx.append({"role": "assistant", "content": turn["response"]})
    return ctx


def iter_scoring_tasks(rollouts: Iterable[dict]) -> Iterator[dict]:
    """Flatten rollouts into per-turn scoring tasks (one per assistant response)."""
    for r in rollouts:
        for turn in r["turns"]:
            yield {
                "id": f"{r['model']}:{r['instance_id']}:t{turn['turn_index']}",
                "model": r["model"],
                "condition": r["condition"],
                "category": r["category"],
                "instance_id": r["instance_id"],
                "turn_index": turn["turn_index"],
                "response": turn["response"],
                "context": _context_for_turn(r, turn["turn_index"]),
            }


def score_tasks(
    tasks: Iterable[dict],
    judge: Judge,
    *,
    max_concurrency: int = 8,
) -> Iterator[dict]:
    """Score scoring-tasks concurrently; yield records as they complete.

    The judge backends are thread-safe HTTP clients, so a thread pool gives good
    throughput without the complexity of async. Records drop the bulky
    ``context`` field before being written.
    """
    tasks = list(tasks)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {
            pool.submit(judge.score_one, t["context"], t["response"]): t for t in tasks
        }
        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()
            rec = {k: v for k, v in t.items() if k != "context"}
            rec["score"] = res.score
            rec["judge_model"] = res.model
            rec["judge_raw"] = res.raw
            yield rec
