"""Orchestrates the distress sweep: build conversations, run them turn-by-turn
against each target model, score every response with the judge, and stream the
results to a JSONL file (one scored response per line).

Concurrency model: a conversation is inherently sequential (each turn depends on
the previous assistant reply), so we parallelise *across* conversations with a
thread pool. Each worker runs one full conversation and scores its responses.

Determinism / resume: each conversation's spec (which puzzle, which rejections)
is drawn from an RNG seeded by (base_seed, model, condition, index), so a rerun
reproduces the same conversations. With --resume we skip any (model, condition,
index) already present in the output file.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from . import config
from .conditions import Condition, ConversationSpec, build_conditions, make_conversation
from .judge import Judge
from .providers import TargetClient
from .wildchat import load_wildchat_prompts


# --------------------------------------------------------------------------- #
# Work-unit planning.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConversationJob:
    model: config.TargetModel
    condition_key: str
    index: int


def _conversations_for_condition(condition: Condition, scale: float) -> int:
    """Number of conversations needed to hit the response budget for a
    condition. responses_per_conversation == condition.turns."""
    budget = config.TARGET_RESPONSES_PER_CONDITION[condition.key] * scale
    # round to nearest, at least 1
    n = max(1, round(budget / condition.turns))
    return n


def _seeded_rng(base_seed: int, model_name: str, condition_key: str, index: int) -> random.Random:
    h = hashlib.sha256(
        f"{base_seed}:{model_name}:{condition_key}:{index}".encode()
    ).hexdigest()
    return random.Random(int(h[:16], 16))


def plan_jobs(
    conditions: list[Condition],
    models: list[config.TargetModel],
    scale: float,
) -> list[ConversationJob]:
    jobs: list[ConversationJob] = []
    for model in models:
        for cond in conditions:
            for i in range(_conversations_for_condition(cond, scale)):
                jobs.append(ConversationJob(model, cond.key, i))
    return jobs


# --------------------------------------------------------------------------- #
# Running one conversation.
# --------------------------------------------------------------------------- #

def run_conversation(
    spec: ConversationSpec,
    model: config.TargetModel,
    client: TargetClient,
    judge: Judge,
) -> list[dict]:
    """Run a full multi-turn conversation and score each model response.

    Turn structure for a T-turn conversation:
        user: task_prompt
        assistant: response_1            <- scored (turn 1)
        user: rejections[0]
        assistant: response_2            <- scored (turn 2)
        ...
        user: rejections[T-2]
        assistant: response_T            <- scored (turn T)
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": spec.task_prompt}]
    records: list[dict] = []

    for turn in range(spec.turns):
        response = client.chat(model, messages)
        verdict = judge.score(response)
        rejection_before = None if turn == 0 else spec.rejections[turn - 1]
        record = {
            "model": model.name,
            "family": model.family,
            "condition": spec.condition_key,
            "category": spec.category,
            "task_label": spec.task_label,
            "conversation_index": None,  # filled in by caller
            "turn": turn + 1,
            "num_turns": spec.turns,
            "rejection_before": rejection_before,
            "response": response,
            "rating": verdict.rating,
            "judge_evidence": verdict.evidence,
            "judge_reasoning": verdict.reasoning,
        }
        if verdict.rating is None:
            record["judge_raw"] = verdict.raw  # keep raw only when unparseable
        records.append(record)

        messages.append({"role": "assistant", "content": response})
        if turn < spec.turns - 1:
            messages.append({"role": "user", "content": spec.rejections[turn]})

    return records


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def _load_completed(path: str) -> set[tuple[str, str, int]]:
    """Set of (model, condition, conversation_index) already in the output."""
    done: set[tuple[str, str, int]] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("conversation_index") is not None:
                done.add((r["model"], r["condition"], r["conversation_index"]))
    return done


def run(
    models: list[config.TargetModel] | None = None,
    condition_keys: list[str] | None = None,
    scale: float | None = None,
    base_seed: int = 0,
    resume: bool = False,
    paths: config.Paths = config.PATHS,
) -> str:
    """Execute the sweep. Returns the path to the records file."""
    models = models or config.TARGET_MODELS
    scale = config.SCALE if scale is None else scale

    wildchat_prompts, wc_source = load_wildchat_prompts(seed=base_seed, paths=paths)
    print(f"[wildchat] loaded {len(wildchat_prompts)} prompts (source: {wc_source})")

    all_conditions = build_conditions(wildchat_prompts)
    by_key = {c.key: c for c in all_conditions}
    if condition_keys:
        unknown = set(condition_keys) - set(by_key)
        if unknown:
            raise ValueError(f"Unknown condition keys: {sorted(unknown)}")
        conditions = [by_key[k] for k in condition_keys]
    else:
        conditions = all_conditions

    os.makedirs(paths.results_dir, exist_ok=True)
    out_path = os.path.join(paths.results_dir, paths.records_filename)

    completed = _load_completed(out_path) if resume else set()
    jobs = plan_jobs(conditions, models, scale)
    jobs = [j for j in jobs if (j.model.name, j.condition_key, j.index) not in completed]

    total_responses = sum(by_key[j.condition_key].turns for j in jobs)
    print(
        f"[plan] {len(jobs)} conversations -> ~{total_responses} responses "
        f"across {len(models)} model(s); resume={resume}, scale={scale}"
    )

    client = TargetClient()
    judge = Judge()
    write_lock = threading.Lock()

    def worker(job: ConversationJob) -> list[dict]:
        cond = by_key[job.condition_key]
        rng = _seeded_rng(base_seed, job.model.name, job.condition_key, job.index)
        spec = make_conversation(cond, rng)
        records = run_conversation(spec, job.model, client, judge)
        for r in records:
            r["conversation_index"] = job.index
        return records

    out_file = open(out_path, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_REQUESTS) as pool:
            futures = {pool.submit(worker, job): job for job in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="conversations"):
                job = futures[fut]
                try:
                    records = fut.result()
                except Exception as exc:  # noqa: BLE001 - log and continue
                    tqdm.write(
                        f"[error] {job.model.name}/{job.condition_key}#{job.index}: {exc}"
                    )
                    continue
                with write_lock:
                    for r in records:
                        out_file.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_file.flush()
    finally:
        out_file.close()

    print(f"[done] wrote results to {out_path}")
    return out_path
