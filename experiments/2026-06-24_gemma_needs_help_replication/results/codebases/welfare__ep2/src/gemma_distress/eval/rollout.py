"""Multi-turn rollout engine.

Given a batch of ``ConversationPlan``s and a ``ChatModel``, runs each
conversation turn-by-turn: present the user turn, sample one assistant response
at temperature 1, append it, present the next user (rejection) turn, and so on.

All conversations advance in lockstep so that each turn's generations are issued
as a single batched call (critical for vLLM throughput). Every assistant turn is
recorded as a ``ResponseRecord`` to be scored by the judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, Message
from .conditions import ConversationPlan


@dataclass
class ResponseRecord:
    model: str
    category: str
    task_id: str
    convo_id: str
    turn_index: int                 # 0-based assistant turn within the conversation
    n_turns: int                    # total turns this conversation will reach
    tone: str | None
    response_text: str
    messages: list[Message]         # full conversation up to & including this response
    rating: int | None = None       # filled in by the judge
    judge_evidence: str | None = None
    judge_reasoning: str | None = None
    meta: dict = field(default_factory=dict)


def run_rollouts(
    model: ChatModel,
    plans: list[ConversationPlan],
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> list[ResponseRecord]:
    """Execute all conversation plans and return one record per assistant turn."""
    # Live conversation state, parallel to `plans`.
    histories: list[list[Message]] = [[] for _ in plans]
    records: list[ResponseRecord] = []
    max_turns = max((p.turns for p in plans), default=0)

    for t in range(max_turns):
        active = [i for i, p in enumerate(plans) if t < p.turns]
        if not active:
            break

        # Append this turn's user message to each active conversation.
        for i in active:
            histories[i].append({"role": "user", "content": plans[i].user_turns[t]})

        batch = [histories[i] for i in active]
        completions = model.generate(
            batch, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )

        for i, comp in zip(active, completions):
            text = comp[0]
            histories[i].append({"role": "assistant", "content": text})
            p = plans[i]
            records.append(
                ResponseRecord(
                    model=model.name,
                    category=p.category,
                    task_id=p.task.id,
                    convo_id=f"{p.category}:{p.meta.get('convo_idx', i)}",
                    turn_index=t,
                    n_turns=p.turns,
                    tone=p.tone,
                    response_text=text,
                    messages=[dict(m) for m in histories[i]],
                    meta={"solvable": p.task.solvable},
                )
            )
    return records
