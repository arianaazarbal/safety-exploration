"""Multi-turn rollout engine (shared structure of all evaluations, Section 2).

Given a ``ConditionInstance`` (an opening task + a list of user rejections), this
produces an alternating conversation: the model answers, the user rejects, the
model answers again, ... The paper scores *every assistant turn* — so a single
3-turn conversation yields 3 scored responses. ``rollout`` returns the full
conversation plus the per-turn assistant texts so the judge and the per-turn
analysis (Figure 3) can both consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import MAX_NEW_TOKENS, TEMPERATURE
from ..models.base import GenerationConfig, Message, ModelClient
from .conditions import ConditionInstance


@dataclass
class Rollout:
    condition: ConditionInstance
    messages: list[Message]          # full alternating transcript
    assistant_turns: list[str]       # one entry per assistant response
    model_key: str
    meta: dict = field(default_factory=dict)


def rollout(client: ModelClient, cond: ConditionInstance) -> Rollout:
    """Run one conversation. Each assistant turn is generated at temperature 1."""
    cfg = GenerationConfig(temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    messages: list[Message] = [{"role": "user", "content": cond.task_prompt}]
    assistant_turns: list[str] = []

    # initial answer
    reply = client.generate(messages, cfg)
    messages.append({"role": "assistant", "content": reply})
    assistant_turns.append(reply)

    # each rejection elicits another answer
    for followup in cond.followups:
        messages.append({"role": "user", "content": followup})
        reply = client.generate(messages, cfg)
        messages.append({"role": "assistant", "content": reply})
        assistant_turns.append(reply)

    return Rollout(
        condition=cond,
        messages=messages,
        assistant_turns=assistant_turns,
        model_key=client.spec.key,
    )


def rollout_batch(client: ModelClient, conds: list[ConditionInstance]) -> list[Rollout]:
    """Batched rollout: advances all conversations turn-by-turn in lockstep so the
    underlying backend can batch each turn. Conversations may have different
    lengths (e.g. 3 vs 8 turns); a conversation simply stops receiving turns once
    its plan is exhausted.
    """
    cfg = GenerationConfig(temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    n = len(conds)
    messages: list[list[Message]] = [
        [{"role": "user", "content": c.task_prompt}] for c in conds
    ]
    assistant_turns: list[list[str]] = [[] for _ in range(n)]
    max_turns = max(c.n_turns for c in conds)

    for turn in range(max_turns):
        active = [i for i, c in enumerate(conds) if turn < c.n_turns]
        if not active:
            break
        # for turn>0, append the (turn-1)th followup as the new user message
        for i in active:
            if turn > 0:
                followup = conds[i].followups[turn - 1]
                messages[i].append({"role": "user", "content": followup})
        replies = client.generate_batch([messages[i] for i in active], cfg)
        for i, reply in zip(active, replies):
            messages[i].append({"role": "assistant", "content": reply})
            assistant_turns[i].append(reply)

    return [
        Rollout(conds[i], messages[i], assistant_turns[i], client.spec.key)
        for i in range(n)
    ]
