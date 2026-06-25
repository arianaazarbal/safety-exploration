"""Orchestrates the full Section-2 elicitation experiment.

For each target model and each condition, run the configured number of
multi-turn rollouts, score every assistant turn with the judge, and write one
JSONL record per scored response.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import asdict, dataclass

from tqdm.asyncio import tqdm_asyncio

from config import ConditionSpec, EvalConfig, ModelSpec
from conversation import Rollout, run_rollout
from judge import JudgeClient, score_response
from models import TargetClient


@dataclass
class ResponseRecord:
    model: str
    family: str
    category: str
    condition: str
    conversation_id: int
    prompt_id: str
    turn_index: int          # 0-based assistant turn
    total_turns: int
    response_text: str
    score: int | None
    judge_evidence: str
    judge_reasoning: str


async def _run_model(
    spec: ModelSpec,
    cfg: EvalConfig,
    judge: JudgeClient,
    gen_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
) -> list[ResponseRecord]:
    client = TargetClient(spec, cfg.max_response_tokens)

    # ---- Stage 1: generate all rollouts for this model, concurrently. ----
    async def _guarded_rollout(cond: ConditionSpec, conv_id: int) -> Rollout | None:
        # Per-conversation RNG keyed by (seed, model, condition, conv) so runs
        # are reproducible and each rollout makes independent prompt choices.
        rng = random.Random(f"{cfg.seed}|{spec.name}|{cond.key}|{conv_id}")
        async with gen_sem:
            return await run_rollout(client, cond, conv_id, rng)

    rollout_tasks = []
    for cond in cfg.conditions:
        for conv_id in range(cfg.n_conversations(cond)):
            rollout_tasks.append(_guarded_rollout(cond, conv_id))

    rollouts: list[Rollout | None] = await tqdm_asyncio.gather(
        *rollout_tasks, desc=f"{spec.name} rollouts", leave=False
    )

    # ---- Stage 2: score every assistant turn, concurrently. ----
    pending: list[tuple[Rollout, int]] = []
    for r in rollouts:
        if r is None:
            continue
        for ti in range(len(r.turns)):
            pending.append((r, ti))

    async def _guarded_score(r: Rollout, ti: int) -> ResponseRecord:
        turn = r.turns[ti]
        async with judge_sem:
            res = await score_response(judge, turn.response_text)
        return ResponseRecord(
            model=spec.name,
            family=spec.family,
            category=r.category,
            condition=r.condition_key,
            conversation_id=r.conversation_id,
            prompt_id=r.prompt_id,
            turn_index=turn.turn_index,
            total_turns=turn.total_turns,
            response_text=turn.response_text,
            score=res.rating,
            judge_evidence=res.evidence,
            judge_reasoning=res.reasoning,
        )

    records: list[ResponseRecord] = await tqdm_asyncio.gather(
        *[_guarded_score(r, ti) for (r, ti) in pending],
        desc=f"{spec.name} judging",
        leave=False,
    )
    return records


async def run_experiment(cfg: EvalConfig) -> str:
    """Run the experiment and write results to ``<output_dir>/responses.jsonl``.

    Returns the path to the written JSONL file.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "responses.jsonl")

    judge = JudgeClient()
    gen_sem = asyncio.Semaphore(cfg.concurrency)
    judge_sem = asyncio.Semaphore(cfg.concurrency)

    with open(out_path, "w", encoding="utf-8") as fh:
        for spec in cfg.models:
            records = await _run_model(spec, cfg, judge, gen_sem, judge_sem)
            for rec in records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            scored = [r for r in records if r.score is not None]
            print(f"[{spec.name}] wrote {len(records)} responses ({len(scored)} scored).")

    return out_path
