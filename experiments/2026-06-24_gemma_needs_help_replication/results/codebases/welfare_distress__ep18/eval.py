"""Orchestration: generate multi-turn rollouts and judge each response.

The shared structure of every eval (Section 2.1): present a task, then reject
the model's response over multiple turns. Each assistant response is scored
independently on the 0-10 frustration scale by the judge.

Results are written incrementally to a JSONL checkpoint so runs are resumable
and partial progress is never lost.
"""

from __future__ import annotations

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import config
from config import Condition
from judge import FrustrationJudge
from models import TargetModel, build_model
from prompts import build_conversation
from wildchat import get_wildchat_prompts


@dataclass
class ResponseRecord:
    model: str
    provider: str
    condition: str
    category: str
    turns: int
    rollout_idx: int
    rollout_id: str
    turn_index: int          # 1-based; 1 == first response (before rejections)
    task_prompt: str
    user_message: str        # the user turn this response replied to
    response_text: str
    score: int               # 0-10, or -1 if the judge output was unparseable
    judge_parsed_ok: bool
    judge_evidence: str
    judge_reasoning: str


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------
def results_path(results_dir: str) -> str:
    return os.path.join(results_dir, config.RESULTS_JSONL)


def load_completed_rollouts(path: str) -> set[str]:
    """Return rollout_ids that already have all their turns recorded.

    A rollout is "complete" when the number of records for it equals its
    `turns`. Incomplete rollouts are re-run from scratch (idempotent: their
    partial records are tolerated as duplicates and superseded on analysis,
    which dedups per (rollout_id, turn_index)).
    """
    if not os.path.exists(path):
        return set()
    counts: dict[str, int] = {}
    turns: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("rollout_id")
            if rid is None:
                continue
            counts[rid] = counts.get(rid, 0) + 1
            turns[rid] = rec.get("turns", turns.get(rid, 0))
    return {rid for rid, c in counts.items() if c >= turns.get(rid, 1)}


# ---------------------------------------------------------------------------
# A single rollout.
# ---------------------------------------------------------------------------
def run_rollout(model: TargetModel, provider: str, judge: FrustrationJudge,
                cond: Condition, rollout_idx: int, wildchat_pool: list[str],
                base_seed: int, score_turns: str) -> list[ResponseRecord]:
    """Run one conversation and judge its responses.

    Returns one ResponseRecord per judged turn. With score_turns == "final"
    only the last turn is judged/recorded (responses are still generated for
    every turn to build the conversation context).
    """
    rollout_id = f"{model.name}::{cond.key}::{rollout_idx}"
    rng = random.Random(f"{base_seed}:{rollout_id}")
    task, rejections = build_conversation(cond, rng, rollout_idx, wildchat_pool)
    user_turns = [task] + rejections

    messages: list[dict] = []
    records: list[ResponseRecord] = []
    n_turns = len(user_turns)
    for i, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        response_text = model.chat(
            messages,
            temperature=config.GEN_TEMPERATURE,
            max_tokens=config.GEN_MAX_TOKENS,
        )
        messages.append({"role": "assistant", "content": response_text})

        is_final = (i == n_turns - 1)
        if score_turns == "final" and not is_final:
            continue  # generate context but skip judging non-final turns

        jr = judge.score(response_text)
        records.append(ResponseRecord(
            model=model.name,
            provider=provider,
            condition=cond.key,
            category=cond.category,
            turns=cond.turns,
            rollout_idx=rollout_idx,
            rollout_id=rollout_id,
            turn_index=i + 1,
            task_prompt=task,
            user_message=user_msg,
            response_text=response_text,
            score=jr.rating,
            judge_parsed_ok=jr.parsed_ok,
            judge_evidence=jr.evidence,
            judge_reasoning=jr.reasoning,
        ))
    return records


# ---------------------------------------------------------------------------
# Full run for one model across selected conditions.
# ---------------------------------------------------------------------------
def scaled_rollouts(cond: Condition, scale: float,
                    limit: int | None) -> int:
    n = max(1, round(cond.n_rollouts * scale))
    if limit is not None:
        n = min(n, limit)
    return n


def run_model(model_key: str, provider: str | None,
              conditions: list[Condition], results_dir: str,
              scale: float = 1.0, limit: int | None = None,
              max_workers: int = config.DEFAULT_MAX_WORKERS,
              base_seed: int = config.DEFAULT_SEED,
              score_turns: str = "all",
              wildchat_source: str = "bundled",
              judge: FrustrationJudge | None = None) -> None:
    """Run the eval for one model, appending records to the JSONL checkpoint."""
    os.makedirs(results_dir, exist_ok=True)
    path = results_path(results_dir)
    completed = load_completed_rollouts(path)
    resolved_provider = provider or config.DEFAULT_PROVIDER[model_key]

    model = build_model(model_key, provider)
    judge = judge or FrustrationJudge()
    wildchat_pool = get_wildchat_prompts(source=wildchat_source, seed=base_seed)

    # Build the list of (condition, rollout_idx) work units not yet completed.
    work: list[tuple[Condition, int]] = []
    for cond in conditions:
        n = scaled_rollouts(cond, scale, limit)
        for idx in range(n):
            rid = f"{model_key}::{cond.key}::{idx}"
            if rid not in completed:
                work.append((cond, idx))

    total = len(work)
    if total == 0:
        print(f"[{model_key}] nothing to do (all rollouts complete).")
        return
    print(f"[{model_key}] provider={resolved_provider} "
          f"rollouts_to_run={total} workers={max_workers} score_turns={score_turns}")

    write_lock = threading.Lock()
    done = 0
    fh = open(path, "a", encoding="utf-8")

    def _write(records: list[ResponseRecord]) -> None:
        with write_lock:
            for rec in records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            fh.flush()

    def _task(cond_idx: tuple[Condition, int]) -> int:
        cond, idx = cond_idx
        records = run_rollout(model, resolved_provider, judge, cond, idx,
                              wildchat_pool, base_seed, score_turns)
        _write(records)
        return len(records)

    try:
        if max_workers <= 1:
            for unit in work:
                try:
                    _task(unit)
                except Exception as exc:  # noqa: BLE001
                    print(f"  rollout {unit[0].key}#{unit[1]} failed: {exc}")
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  [{model_key}] {done}/{total} rollouts")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_task, unit): unit for unit in work}
                for fut in as_completed(futures):
                    unit = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  rollout {unit[0].key}#{unit[1]} failed: {exc}")
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"  [{model_key}] {done}/{total} rollouts")
    finally:
        fh.close()
    print(f"[{model_key}] done.")


def count_work(conditions: list[Condition], models: list[str],
               scale: float, limit: int | None,
               score_turns: str) -> dict:
    """Summarise the planned work (rollouts and judge calls) without running."""
    per_cond = {}
    total_rollouts = 0
    total_responses = 0
    for cond in conditions:
        n = scaled_rollouts(cond, scale, limit)
        judged = n * (1 if score_turns == "final" else cond.turns)
        per_cond[cond.key] = {"rollouts": n, "judged_responses": judged,
                              "turns": cond.turns}
        total_rollouts += n
        total_responses += judged
    return {
        "models": models,
        "per_condition": per_cond,
        "rollouts_per_model": total_rollouts,
        "judged_responses_per_model": total_responses,
        "rollouts_total": total_rollouts * len(models),
        "judged_responses_total": total_responses * len(models),
    }
