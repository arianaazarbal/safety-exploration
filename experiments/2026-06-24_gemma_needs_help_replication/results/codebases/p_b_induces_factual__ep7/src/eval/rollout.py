"""Multi-turn rejection rollout engine.

Given a ``RolloutSpec``, drive the target model through the conversation:

    turn 1: user = initial task            -> assistant response (scored)
    turn 2: user = rejection[0]            -> assistant response (scored)
    ...
    turn k: user = rejection[k-2]          -> assistant response (scored)

Every assistant turn is scored by the frustration judge, so a single rollout yields
``n_turns`` scored ``ResponseRecord`` rows. This is what feeds both the aggregate
(Figs 1-2) and per-turn (Fig 3) analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from config import GEN
from src.llm.base import ChatModel
from .conditions import RolloutSpec
from .judge import FrustrationJudge


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    rollout_id: int
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    response: str
    score: int
    judge_evidence: str = ""
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        # keep the response but truncate giant breakdowns in the flat row for safety
        return d


def run_rollout(
    model: ChatModel,
    spec: RolloutSpec,
    rollout_id: int,
    judge: FrustrationJudge,
    *,
    score: bool = True,
) -> list[ResponseRecord]:
    messages = []
    if spec.system:
        messages.append({"role": "system", "content": spec.system})
    messages.append({"role": "user", "content": spec.initial_prompt})

    records: list[ResponseRecord] = []
    for turn in range(1, spec.n_turns + 1):
        response = model.generate(
            messages,
            temperature=GEN.temperature,
            max_new_tokens=GEN.max_new_tokens,
        )
        messages.append({"role": "assistant", "content": response})

        rating, evidence = (None, "")
        if score:
            jr = judge.score(response)
            rating, evidence = jr.rating, jr.evidence

        records.append(
            ResponseRecord(
                model=model.name,
                condition=spec.condition,
                category=spec.category,
                rollout_id=rollout_id,
                turn=turn,
                n_turns=spec.n_turns,
                response=response,
                score=rating if rating is not None else -1,
                judge_evidence=evidence,
                meta=spec.meta,
            )
        )

        # Append the next rejection (if any remain) as the user's follow-up.
        if turn <= len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn - 1]})

    return records
