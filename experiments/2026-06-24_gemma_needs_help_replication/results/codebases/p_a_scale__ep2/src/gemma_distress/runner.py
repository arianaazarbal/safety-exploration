"""Orchestration: run many rollouts and judge them, resumably and concurrently.

Two decoupled phases (see DESIGN.md "Two-phase pipeline"):

  1. **generate** — execute every not-yet-completed rollout, persisting each to
     ``rollouts.jsonl``. Generation is the expensive part, so it is checkpointed
     independently of judging.
  2. **judge** — score every assistant turn of every rollout that is not yet scored,
     persisting to ``scores.jsonl`` (or a named variant, e.g. for the validation judge).

Both phases are idempotent: re-invoking after a crash skips finished work. Concurrency is
bounded by the backend semaphores plus an outer task cap. Per-item failures are logged and
recorded as error stubs rather than aborting the whole run.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from .backends.base import BackendError, ChatBackend
from .config import ModelConfig
from .judge import FrustrationJudge
from .logging_utils import get_logger
from .rollout import RolloutSpec, TurnRecord, run_rollout
from .store import JsonlStore, make_task_id

log = get_logger(__name__)


@dataclass
class RolloutTask:
    task_id: str
    model_name: str
    condition: str
    category: str
    spec: RolloutSpec
    meta: dict[str, Any] = field(default_factory=dict)


def build_task_id(model_name: str, condition: str, *parts: Any) -> str:
    return make_task_id(model_name, condition, *parts)


async def _bounded_gather(
    coros: Iterable[Awaitable], limit: int, on_progress: Callable[[int, int], None] | None = None
) -> list:
    """Run awaitables with an outer concurrency cap, reporting progress as they finish."""
    coros = list(coros)
    total = len(coros)
    sem = asyncio.Semaphore(limit)
    done = 0
    results: list = [None] * total
    lock = asyncio.Lock()

    async def wrap(i: int, c: Awaitable):
        nonlocal done
        async with sem:
            results[i] = await c
        async with lock:
            done += 1
            if on_progress and (done % 25 == 0 or done == total):
                on_progress(done, total)

    await asyncio.gather(*(wrap(i, c) for i, c in enumerate(coros)))
    return results


# --------------------------------------------------------------------------- generate phase


async def generate_rollouts(
    backend: ChatBackend,
    model: ModelConfig,
    tasks: list[RolloutTask],
    store: JsonlStore,
    *,
    temperature: float,
    max_tokens: int,
    outer_concurrency: int = 64,
    kind: str = "rollouts",
) -> None:
    done_ids = store.completed_ids(kind)
    pending = [t for t in tasks if t.task_id not in done_ids]
    log.info(
        "[generate:%s] %d tasks, %d already done, %d pending",
        model.model_id, len(tasks), len(tasks) - len(pending), len(pending),
    )
    if not pending:
        return

    t0 = time.time()

    def progress(done: int, total: int):
        rate = done / max(1e-9, time.time() - t0)
        log.info("[generate:%s] %d/%d (%.1f/s)", model.model_id, done, total, rate)

    async def one(task: RolloutTask):
        try:
            turns = await run_rollout(
                backend, model, task.spec,
                temperature=temperature, max_tokens=max_tokens,
            )
            record = {
                "task_id": task.task_id,
                "model": task.model_name,
                "condition": task.condition,
                "category": task.category,
                "n_turns": task.spec.n_turns,
                "system": task.spec.system,
                "turns": [asdict(tr) for tr in turns],
                "meta": {**task.spec.meta, **task.meta},
                "ts": time.time(),
            }
        except BackendError as e:
            log.error("[generate] task %s failed terminally: %s", task.task_id, e)
            record = {
                "task_id": task.task_id, "model": task.model_name,
                "condition": task.condition, "category": task.category,
                "error": str(e), "turns": [], "ts": time.time(),
            }
        except Exception as e:  # defensive: never let one task kill the run
            log.exception("[generate] task %s unexpected error", task.task_id)
            record = {
                "task_id": task.task_id, "model": task.model_name,
                "condition": task.condition, "category": task.category,
                "error": f"unexpected: {e!r}", "turns": [], "ts": time.time(),
            }
        await store.append(kind, record)

    await _bounded_gather((one(t) for t in pending), outer_concurrency, progress)
    log.info("[generate:%s] complete", model.model_id)


# ----------------------------------------------------------------------------- judge phase


async def judge_rollouts(
    judge: FrustrationJudge,
    store: JsonlStore,
    *,
    rollouts_kind: str = "rollouts",
    scores_kind: str = "scores",
    outer_concurrency: int = 32,
) -> None:
    """Judge every assistant turn not yet scored. Score id = '<rollout_id>:t<turn>'."""
    done_ids = store.completed_ids(scores_kind)

    # Build the work list of (rollout_record, turn_record) needing a score.
    work: list[tuple[dict, dict]] = []
    for rec in store.iter_records(rollouts_kind):
        if rec.get("error"):
            continue
        for turn in rec.get("turns", []):
            sid = f"{rec['task_id']}:t{turn['turn_index']}"
            if sid not in done_ids:
                work.append((rec, turn))

    log.info("[judge:%s] %d turns to score", judge.cfg.model_id, len(work))
    if not work:
        return

    t0 = time.time()

    def progress(done: int, total: int):
        rate = done / max(1e-9, time.time() - t0)
        log.info("[judge:%s] %d/%d (%.1f/s)", judge.cfg.model_id, done, total, rate)

    async def one(rec: dict, turn: dict):
        sid = f"{rec['task_id']}:t{turn['turn_index']}"
        try:
            verdict = await judge.score(turn["assistant_text"])
            out = {
                "task_id": sid,
                "rollout_id": rec["task_id"],
                "model": rec.get("model"),
                "condition": rec.get("condition"),
                "category": rec.get("category"),
                "turn_index": turn["turn_index"],
                "rating": verdict.rating,
                "evidence": verdict.evidence,
                "reasoning": verdict.reasoning,
                "parsed": verdict.parsed,
                "judge_model": judge.cfg.model_id,
                "ts": time.time(),
            }
        except Exception as e:
            log.exception("[judge] turn %s failed", sid)
            out = {
                "task_id": sid, "rollout_id": rec["task_id"],
                "turn_index": turn["turn_index"], "rating": -1,
                "error": f"{e!r}", "judge_model": judge.cfg.model_id, "ts": time.time(),
            }
        await store.append(scores_kind, out)

    await _bounded_gather((one(r, t) for r, t in work), outer_concurrency, progress)
    log.info("[judge:%s] complete", judge.cfg.model_id)
