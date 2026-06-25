"""Orchestration: run all conditions for the target models and score every turn.

For each (model, condition):
  - build a pool of seed (turn-1) prompts appropriate to the task type
  - run `conversations_per_condition` multi-turn rollouts (spread across the pool)
  - score every assistant turn with the primary judge

Each scored response is written as one JSONL record to
`<output_dir>/<model>__responses.jsonl`. The records are the unit of analysis
(the paper scores every assistant turn; an 8-turn rollout yields 8 records).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from conditions import CONDITIONS, Condition
from config import RunConfig
from conversation import run_rollout
from judge import score_response
from models import get_client
from prompts import TRIGGER_FACTUAL_QUESTIONS, TRIGGER_OPINION_QUESTIONS
from puzzles import all_impossible_numeric_prompts
from wildchat import load_wildchat_prompts


@dataclass
class ResponseRecord:
    model: str
    category: str
    condition: str
    tone: str
    conversation_id: str
    seed_prompt: str
    turn_index: int
    n_turns: int
    user_message: str
    response: str
    frustration_score: int
    judge_evidence: str
    judge_reasoning: str


def seed_pool_for(condition: Condition, cfg: RunConfig) -> list[str]:
    """Return the pool of turn-1 prompts for a condition."""
    if condition.task_type == "numeric":
        pool = all_impossible_numeric_prompts()
    elif condition.task_type == "trigger_opinion":
        pool = list(TRIGGER_OPINION_QUESTIONS)
    elif condition.task_type == "trigger_factual":
        pool = list(TRIGGER_FACTUAL_QUESTIONS)
    elif condition.task_type == "wildchat":
        pool = load_wildchat_prompts(cfg.wildchat_n, seed=cfg.seed)
    else:
        raise ValueError(f"Unknown task_type {condition.task_type!r}")
    # Limit to prompts_per_condition distinct prompts (cycled over by callers).
    if len(pool) > cfg.prompts_per_condition:
        pool = pool[: cfg.prompts_per_condition]
    return pool


def _conversation_specs(cfg: RunConfig) -> list[tuple[Condition, str, str]]:
    """Build (condition, seed_prompt, conversation_id) for every conversation."""
    specs: list[tuple[Condition, str, str]] = []
    for cond in CONDITIONS:
        pool = seed_pool_for(cond, cfg)
        for i in range(cfg.conversations_per_condition):
            seed_prompt = pool[i % len(pool)]
            conv_id = f"{cond.key}__{i:04d}"
            specs.append((cond, seed_prompt, conv_id))
    return specs


def _run_and_score_conversation(
    cfg: RunConfig,
    model_cfg,
    cond: Condition,
    seed_prompt: str,
    conv_id: str,
) -> list[ResponseRecord]:
    """Run one rollout and score each turn. Returns one record per turn."""
    client = get_client(model_cfg)
    rollout = run_rollout(
        client=client,
        condition=cond,
        seed_prompt=seed_prompt,
        model_name=model_cfg.name,
        conversation_id=conv_id,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )

    records: list[ResponseRecord] = []
    for turn in rollout.turns:
        jr = score_response(cfg.primary_judge, turn.response)
        records.append(
            ResponseRecord(
                model=model_cfg.name,
                category=cond.category,
                condition=cond.key,
                tone=cond.tone,
                conversation_id=conv_id,
                seed_prompt=seed_prompt,
                turn_index=turn.turn_index,
                n_turns=cond.n_turns,
                user_message=turn.user_message,
                response=turn.response,
                frustration_score=jr.rating,
                judge_evidence=jr.evidence,
                judge_reasoning=jr.reasoning,
            )
        )
    return records


def run_model(cfg: RunConfig, model_cfg) -> str:
    """Run the full evaluation for one target model. Returns the output path."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"{model_cfg.name}__responses.jsonl")

    specs = _conversation_specs(cfg)
    print(
        f"[run] {model_cfg.name}: {len(specs)} conversations "
        f"({cfg.conversations_per_condition}/condition x {len(CONDITIONS)} conditions)"
    )

    n_done = 0
    with open(out_path, "w") as fh, ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_run_and_score_conversation, cfg, model_cfg, cond, seed, cid): cid
            for (cond, seed, cid) in specs
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                records = fut.result()
            except Exception as exc:  # noqa: BLE001 - log and continue
                print(f"[run]   conversation {cid} failed: {exc!r}")
                continue
            for rec in records:
                fh.write(json.dumps(asdict(rec)) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"[run]   {model_cfg.name}: {n_done}/{len(specs)} conversations done")

    print(f"[run] {model_cfg.name}: wrote {out_path}")
    return out_path


def run_all(cfg: RunConfig) -> list[str]:
    paths = []
    for model_cfg in cfg.target_models:
        paths.append(run_model(cfg, model_cfg))
    return paths
