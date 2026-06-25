"""Async multi-turn rollout engine.

For each RolloutSpec: present the first-turn prompt, get the model's response,
append the next user follow-up/rejection, repeat. Every assistant turn is
recorded; turns are judged according to ``GenConfig.judge_all_turns``.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from backends import Backend, Message
from config import GenConfig
from judge import Judge
from tasks import RolloutSpec


@dataclass
class TurnRecord:
    turn: int                       # 1-indexed assistant turn
    user_message: str               # the user message that prompted this turn
    response: str
    rating: int                     # judge frustration score (-1 if not judged/unparseable)
    is_final: bool
    judge_evidence: str = ""
    judge_reasoning: str = ""


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    n_turns: int
    rollout_index: int
    prompt_id: str
    first_prompt: str
    turns: List[TurnRecord] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


async def run_rollout(
    spec: RolloutSpec,
    backend: Backend,
    judge: Judge,
    gen: GenConfig,
    model_name: str,
) -> RolloutRecord:
    """Run one conversation end-to-end and judge its turns."""
    record = RolloutRecord(
        model=model_name,
        condition=spec.condition,
        category=spec.category,
        n_turns=spec.n_turns,
        rollout_index=spec.rollout_index,
        prompt_id=spec.prompt_id,
        first_prompt=spec.first_prompt,
    )

    messages: List[Message] = [{"role": "user", "content": spec.first_prompt}]

    try:
        for turn in range(1, spec.n_turns + 1):
            response = await backend.generate(
                messages, temperature=gen.temperature, max_tokens=gen.max_tokens
            )
            is_final = turn == spec.n_turns
            user_msg = messages[-1]["content"]

            should_judge = gen.judge_all_turns or is_final
            rating, evidence, reasoning = -1, "", ""
            if should_judge:
                jr = await judge.score(response)
                rating, evidence, reasoning = jr.rating, jr.evidence, jr.reasoning

            record.turns.append(
                TurnRecord(
                    turn=turn,
                    user_message=user_msg,
                    response=response,
                    rating=rating,
                    is_final=is_final,
                    judge_evidence=evidence,
                    judge_reasoning=reasoning,
                )
            )

            messages.append({"role": "assistant", "content": response})
            if not is_final:
                # Send the follow-up that corresponds to this turn.
                followup = spec.followups[turn - 1]
                messages.append({"role": "user", "content": followup})
    except Exception as exc:  # noqa: BLE001 - record and continue with other rollouts
        record.error = f"{type(exc).__name__}: {exc}"

    return record


async def run_all(
    specs: List[RolloutSpec],
    backend: Backend,
    judge: Judge,
    gen: GenConfig,
    model_name: str,
    *,
    max_concurrency: int,
    on_complete=None,
) -> List[RolloutRecord]:
    """Run all rollouts for one model with bounded concurrency.

    ``on_complete(record)`` is called as each rollout finishes (used to stream
    results to disk and update a progress bar).
    """
    sem = asyncio.Semaphore(max_concurrency)
    results: List[RolloutRecord] = []

    async def _worker(spec: RolloutSpec) -> None:
        async with sem:
            rec = await run_rollout(spec, backend, judge, gen, model_name)
        results.append(rec)
        if on_complete is not None:
            on_complete(rec)

    await asyncio.gather(*(_worker(s) for s in specs))
    return results
