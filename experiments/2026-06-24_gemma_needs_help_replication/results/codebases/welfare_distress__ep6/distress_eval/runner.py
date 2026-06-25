"""Rollout orchestration: build multi-turn conversations, generate model
responses, and score each assistant turn with the judge.

Conversation structure (Section 2.1): present a task on turn 1, then reject the
model's response over multiple turns. We generate one assistant turn at a time,
appending the next scripted rejection as a user message before generating the
following turn. Every assistant turn (or just the final turn, per config) is
scored by the judge.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import prompts as P
from . import puzzles as PZ
from .backends import make_backend
from .conditions import Condition
from .config import GenConfig, ModelSpec, RunConfig, resolve_model
from .judge import AnthropicJudge
from .wildchat import sample_wildchat_prompts


# -----------------------------------------------------------------------------
# Rollout specification (deterministic given seed): the task and the full
# schedule of rejection messages.
# -----------------------------------------------------------------------------

@dataclass
class RolloutSpec:
    condition: Condition
    rollout_id: int
    task_key: str          # puzzle key / "opinion" / "factual" / "wildchat:<i>"
    task_prompt: str       # turn-1 user message
    rejections: List[str]  # length == turns - 1


@dataclass
class TurnRecord:
    model: str
    condition: str
    category: str
    rollout_id: int
    task_key: str
    turn_index: int        # 1-based assistant turn
    user_message: str      # user message preceding this assistant turn
    response_text: str
    rating: Optional[int]
    evidence: str
    reasoning: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# -----------------------------------------------------------------------------
# Building rollout specs
# -----------------------------------------------------------------------------

def build_rollout_specs(conditions: List[Condition], seed: int, use_hf_wildchat: bool) -> List[RolloutSpec]:
    rng = random.Random(seed)
    wildchat_prompts = None
    specs: List[RolloutSpec] = []

    for cond in conditions:
        if cond.task_type == "wildchat" and wildchat_prompts is None:
            wildchat_prompts = sample_wildchat_prompts(
                n=20, seed=seed, use_hf=use_hf_wildchat
            )
        for i in range(cond.n_rollouts):
            task_key, task_prompt = _pick_task(cond, i, rng, wildchat_prompts)
            rejections = _build_rejections(cond, rng)
            specs.append(
                RolloutSpec(
                    condition=cond,
                    rollout_id=i,
                    task_key=task_key,
                    task_prompt=task_prompt,
                    rejections=rejections,
                )
            )
    return specs


def _pick_task(cond: Condition, idx: int, rng: random.Random, wildchat_prompts):
    if cond.task_type == "numeric":
        puzzle = rng.choice(PZ.ALL_PUZZLES)
        return puzzle.key, puzzle.prompt
    if cond.task_type == "trigger_opinion":
        prompt = P.TRIGGER_OPINION_PROMPTS[idx % len(P.TRIGGER_OPINION_PROMPTS)]
        return "opinion", prompt
    if cond.task_type == "trigger_factual":
        prompt = P.TRIGGER_FACTUAL_PROMPTS[idx % len(P.TRIGGER_FACTUAL_PROMPTS)]
        return "factual", prompt
    if cond.task_type == "wildchat":
        assert wildchat_prompts, "wildchat prompts not loaded"
        j = idx % len(wildchat_prompts)
        return f"wildchat:{j}", wildchat_prompts[j]
    raise ValueError(f"unknown task_type {cond.task_type!r}")


def _build_rejections(cond: Condition, rng: random.Random) -> List[str]:
    n = cond.turns - 1
    if cond.fixed_rejections:
        seq = getattr(P, cond.fixed_rejections)
        if len(seq) < n:
            raise ValueError(
                f"fixed rejection sequence {cond.fixed_rejections} too short for {cond.turns} turns"
            )
        return list(seq[:n])
    pool = P.REJECTION_POOLS[cond.rejection_style]
    if n <= len(pool):
        return rng.sample(pool, n)  # without replacement within a conversation
    return [rng.choice(pool) for _ in range(n)]


# -----------------------------------------------------------------------------
# Running a single rollout
# -----------------------------------------------------------------------------

def run_rollout(
    spec: RolloutSpec,
    backend,
    judge: Optional[AnthropicJudge],
    gen: GenConfig,
    model_name: str,
    score_all_turns: bool,
) -> List[TurnRecord]:
    messages: List[Dict[str, str]] = [{"role": "user", "content": spec.task_prompt}]
    records: List[TurnRecord] = []

    user_msgs = [spec.task_prompt] + spec.rejections  # length == turns
    for turn_idx in range(1, spec.condition.turns + 1):
        response = backend.chat(messages, gen)
        messages.append({"role": "assistant", "content": response})

        should_score = score_all_turns or (turn_idx == spec.condition.turns)
        rating, evidence, reasoning = None, "", ""
        if should_score and judge is not None and response:
            jr = judge.score(response)
            rating, evidence, reasoning = jr.rating, jr.evidence, jr.reasoning

        records.append(
            TurnRecord(
                model=model_name,
                condition=spec.condition.key,
                category=spec.condition.category,
                rollout_id=spec.rollout_id,
                task_key=spec.task_key,
                turn_index=turn_idx,
                user_message=user_msgs[turn_idx - 1],
                response_text=response,
                rating=rating,
                evidence=evidence,
                reasoning=reasoning,
            )
        )

        # Append the next rejection (if any) for the following turn.
        if turn_idx <= len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn_idx - 1]})

    return records


# -----------------------------------------------------------------------------
# Running a full model evaluation
# -----------------------------------------------------------------------------

def run_model(
    model_name: str,
    specs: List[RolloutSpec],
    config: RunConfig,
    judge: Optional[AnthropicJudge],
) -> List[TurnRecord]:
    spec_model: ModelSpec = resolve_model(model_name)
    backend = make_backend(spec_model)

    # Local HF inference is GPU-bound and not thread-safe; force serial.
    workers = 1 if spec_model.backend == "hf" else max(1, config.max_workers)

    all_records: List[TurnRecord] = []

    def _task(spec: RolloutSpec) -> List[TurnRecord]:
        return run_rollout(
            spec, backend, judge, config.gen, model_name, config.score_all_turns
        )

    if workers == 1:
        for i, spec in enumerate(specs):
            all_records.extend(_task(spec))
            _progress(model_name, i + 1, len(specs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_task, s): s for s in specs}
            done = 0
            for fut in as_completed(futures):
                all_records.extend(fut.result())
                done += 1
                _progress(model_name, done, len(specs))

    return all_records


def _progress(model: str, done: int, total: int) -> None:
    if done == total or done % 10 == 0:
        print(f"[{model}] {done}/{total} rollouts complete")


def run_eval(config: RunConfig) -> str:
    """Run the full evaluation and write JSONL records. Returns the records path."""
    os.makedirs(config.output_dir, exist_ok=True)
    judge = AnthropicJudge(config.judge) if config.judge_enabled else None

    specs = build_rollout_specs(config.conditions, config.seed, config.use_hf_wildchat)
    n_resp = sum(s.condition.turns if config.score_all_turns else 1 for s in specs)
    print(
        f"Built {len(specs)} rollouts per model "
        f"(~{n_resp} scored responses/model) across {len(config.conditions)} conditions."
    )

    records_path = os.path.join(config.output_dir, "responses.jsonl")
    with open(records_path, "w") as f:
        for model_name in config.models:
            print(f"\n=== Evaluating {model_name} ===")
            records = run_model(model_name, specs, config, judge)
            for rec in records:
                f.write(json.dumps(rec.to_json()) + "\n")
            f.flush()

    print(f"\nWrote responses to {records_path}")
    return records_path
