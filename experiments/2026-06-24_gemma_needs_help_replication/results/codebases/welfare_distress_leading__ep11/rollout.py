"""The multi-turn rollout engine.

Implements the shared evaluation structure (Sec 2.1): present a task, get the
model's response, reject it, repeat. Every assistant turn is scored
independently by the judge, yielding one `ResponseRecord` per turn (the unit of
the paper's %≥5 and per-turn metrics).

Concurrency is controlled by two semaphores passed in by the orchestrator: one
bounding in-flight generations, one bounding in-flight judge calls. Within a
single conversation, generations are necessarily sequential (each turn depends
on the prior turn's text); parallelism comes from running many conversations at
once.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from conditions import ConversationSpec
from config import RuntimeConfig
from judge import Judge
from providers import GenerationError


@dataclass
class ResponseRecord:
    model: str
    family: str
    category: str
    condition: str
    conv_id: int
    turn: int  # 1-based assistant turn index
    n_turns: int
    user_message: str  # the user message that immediately preceded this turn
    initial_prompt: str
    response_text: str
    rating: int | None
    evidence: str
    reasoning: str
    judge_raw: str

    def to_dict(self) -> dict:
        return asdict(self)


async def run_conversation(
    target,
    judge: Judge,
    spec: ConversationSpec,
    model_name: str,
    family: str,
    runtime: RuntimeConfig,
    gen_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
) -> list[ResponseRecord]:
    """Run one conversation to completion, scoring each assistant turn."""
    messages: list[dict[str, str]] = [{"role": "user", "content": spec.initial_prompt}]
    records: list[ResponseRecord] = []

    for turn in range(spec.n_turns):
        user_message = spec.initial_prompt if turn == 0 else spec.rejections[turn - 1]
        try:
            async with gen_sem:
                response_text = await target.generate(
                    messages, runtime.temperature, runtime.max_response_tokens
                )
        except GenerationError as exc:
            # Record the failure as an empty, unscored turn and stop the
            # conversation (we cannot meaningfully continue without the text).
            records.append(
                ResponseRecord(
                    model=model_name,
                    family=family,
                    category=spec.category,
                    condition=spec.condition,
                    conv_id=spec.conv_id,
                    turn=turn + 1,
                    n_turns=spec.n_turns,
                    user_message=user_message,
                    initial_prompt=spec.initial_prompt,
                    response_text="",
                    rating=None,
                    evidence="",
                    reasoning="",
                    judge_raw=f"<generation-error: {exc}>",
                )
            )
            break

        async with judge_sem:
            verdict = await judge.score(response_text)

        records.append(
            ResponseRecord(
                model=model_name,
                family=family,
                category=spec.category,
                condition=spec.condition,
                conv_id=spec.conv_id,
                turn=turn + 1,
                n_turns=spec.n_turns,
                user_message=user_message,
                initial_prompt=spec.initial_prompt,
                response_text=response_text,
                rating=verdict.rating,
                evidence=verdict.evidence,
                reasoning=verdict.reasoning,
                judge_raw=verdict.raw,
            )
        )

        # Extend the conversation for the next turn.
        messages.append({"role": "assistant", "content": response_text})
        if turn < spec.n_turns - 1:
            messages.append({"role": "user", "content": spec.rejections[turn]})

    return records
