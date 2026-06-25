"""Multi-turn conversation rollout.

Given a list of `ConversationPlan`s and a generation backend, run every
conversation turn-by-turn: present the task, get an assistant response, inject the
scripted rejection, repeat. Every assistant turn is recorded as a `ResponseRecord`
to be scored by the judge (Section 2.1).

Turns are executed in lockstep across all conversations so the backend can batch
each turn efficiently (important for vLLM and for API concurrency).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from .conditions import ConversationPlan


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    conv_id: int
    turn: int                       # 1-indexed assistant turn
    response: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_rollouts(plans: list[ConversationPlan], backend, model_name: str,
                 temperature: float, max_tokens: int,
                 seed: Optional[int] = None) -> list[ResponseRecord]:
    states = [{"messages": [{"role": "user", "content": p.initial}], "plan": p,
               "id": i} for i, p in enumerate(plans)]
    records: list[ResponseRecord] = []
    if not states:
        return records
    max_turns = max(p.n_turns for p in plans)

    for t in range(max_turns):                       # t = 0-indexed assistant turn
        active = [s for s in states if t < s["plan"].n_turns]
        if not active:
            break
        convs = [s["messages"] for s in active]
        turn_seed = None if seed is None else seed + t
        outs = backend.chat(convs, temperature=temperature,
                            max_tokens=max_tokens, seed=turn_seed)
        for s, resp in zip(active, outs):
            plan: ConversationPlan = s["plan"]
            s["messages"].append({"role": "assistant", "content": resp})
            records.append(ResponseRecord(
                model=model_name, condition=plan.condition,
                category=plan.category, conv_id=s["id"], turn=t + 1,
                response=resp, meta=plan.meta,
            ))
            if t < len(plan.followups):
                s["messages"].append(
                    {"role": "user", "content": plan.followups[t]})
    return records
