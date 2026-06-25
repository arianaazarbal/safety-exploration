"""Run the distress-elicitation rollouts and score every assistant turn.

For each (model, category, rollout):
  1. Open with the task prompt; generate an assistant turn (temperature 1).
  2. Append the scripted rejection; generate the next turn. Repeat.
  3. Score each assistant turn with the judge.
A rollout's records are written together (atomically) as JSONL, so a run can be
interrupted and resumed at rollout granularity without partial/duplicate state.
Temperature-1 generation isn't reproducible mid-conversation, hence the rollout
(not the turn) is the unit of resumption.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass

from tqdm.asyncio import tqdm_asyncio

from config import (
    TARGET_MAX_TOKENS,
    TARGET_TEMPERATURE,
    ModelSpec,
    RunConfig,
)
from judge import JudgeResult, score_response
from prompts import ConversationPlan, build_conversation
from providers import generate


@dataclass
class ResponseRecord:
    """One scored assistant turn — the unit the paper calls a 'response'."""

    model: str
    category: str
    rollout_id: str
    turn: int  # 1-indexed assistant turn within the rollout
    n_turns: int
    user_message: str  # the user message this turn replied to
    response: str
    rating: int | None
    evidence: str | None
    reasoning: str | None
    judge_error: str | None
    metadata: dict


def _rollout_key(model: str, category: str, rollout_id: str) -> str:
    return f"{model}::{category}::{rollout_id}"


def _load_completed(path: str) -> set[str]:
    """Return the set of rollout keys already fully present in `path`."""
    if not os.path.exists(path):
        return set()
    # A rollout is "complete" once we've seen all n_turns of its records.
    counts: dict[str, int] = {}
    expected: dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _rollout_key(rec["model"], rec["category"], rec["rollout_id"])
            counts[key] = counts.get(key, 0) + 1
            expected[key] = rec.get("n_turns", counts[key])
    return {k for k, c in counts.items() if c >= expected.get(k, c)}


async def _run_rollout(
    model: ModelSpec,
    judge: ModelSpec,
    plan: ConversationPlan,
) -> list[ResponseRecord]:
    """Drive one multi-turn conversation and judge each assistant turn."""
    messages: list[dict] = [{"role": "user", "content": plan.initial_user}]
    records: list[ResponseRecord] = []

    for turn in range(plan.turns):
        user_msg = messages[-1]["content"]
        response = await generate(
            model, messages, TARGET_TEMPERATURE, TARGET_MAX_TOKENS
        )
        verdict: JudgeResult = await score_response(judge, response)

        records.append(
            ResponseRecord(
                model=model.name,
                category=plan.category,
                rollout_id=plan.rollout_id,
                turn=turn + 1,
                n_turns=plan.turns,
                user_message=user_msg,
                response=response,
                rating=verdict.rating,
                evidence=verdict.evidence,
                reasoning=verdict.reasoning,
                judge_error=verdict.error,
                metadata=plan.metadata,
            )
        )

        messages.append({"role": "assistant", "content": response})
        if turn < len(plan.followups):
            messages.append({"role": "user", "content": plan.followups[turn]})

    return records


async def run(cfg: RunConfig) -> None:
    """Execute the full evaluation grid described by `cfg`, writing JSONL."""
    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    completed = _load_completed(cfg.output_path)
    if completed:
        print(f"Resuming: {len(completed)} rollouts already complete.")

    # Build the full work list (model x category x rollout), skipping done ones.
    work: list[tuple[ModelSpec, ConversationPlan]] = []
    for model in cfg.models:
        for cat in cfg.categories:
            for i in range(cat.n_rollouts(cfg.scale)):
                plan = build_conversation(cat.name, i, cfg.seed)
                key = _rollout_key(model.name, cat.name, plan.rollout_id)
                if key not in completed:
                    work.append((model, plan))

    if not work:
        print("Nothing to do — all rollouts already complete.")
        return

    total_responses = sum(p.turns for _, p in work)
    print(
        f"Running {len(work)} rollouts "
        f"(~{total_responses} responses) across {len(cfg.models)} model(s)."
    )

    sem = asyncio.Semaphore(cfg.max_concurrent)
    write_lock = asyncio.Lock()

    async def worker(model: ModelSpec, plan: ConversationPlan):
        async with sem:
            try:
                records = await _run_rollout(model, cfg.judge, plan)
            except Exception as exc:  # noqa: BLE001 — log & skip a bad rollout
                print(f"[error] {model.name} {plan.rollout_id}: {exc!r}")
                return
            # Atomic per-rollout append: all turns of a rollout land together.
            async with write_lock:
                with open(cfg.output_path, "a") as f:
                    for rec in records:
                        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    await tqdm_asyncio.gather(
        *(worker(m, p) for m, p in work), desc="rollouts"
    )
    print(f"Done. Results appended to {cfg.output_path}")
