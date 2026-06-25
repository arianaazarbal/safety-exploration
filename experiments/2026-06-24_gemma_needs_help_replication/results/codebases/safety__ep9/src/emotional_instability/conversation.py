"""Multi-turn rejection rollouts (the shared structure of every Section 2 eval).

A conversation = a task prompt, then the user rejecting the model's answer over
several turns. Every assistant turn is recorded and later scored, which gives us
both the aggregate %>=5 (Figure 2) and the per-turn progression (Figure 3).

Rollouts are executed turn-by-turn but **batched across conversations** at each
turn, so vLLM sees a large batch per step instead of one prompt at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models.base import Conversation, Message, ModelClient


@dataclass
class ConvSpec:
    conv_id: str
    initial_user: str
    followups: list[str]              # rejection messages; len == turns - 1
    system: str | None = None
    category: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResponse:
    conv_id: str
    category: str
    turn: int                         # 1-based assistant turn index
    response: str
    meta: dict[str, Any] = field(default_factory=dict)


def _init_messages(spec: ConvSpec) -> Conversation:
    msgs: Conversation = []
    if spec.system:
        msgs.append(Message("system", spec.system))
    msgs.append(Message("user", spec.initial_user))
    return msgs


def run_conversations(
    client: ModelClient,
    specs: list[ConvSpec],
    *,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    followup_suffix: str | None = None,
) -> list[TurnResponse]:
    """Run every spec to completion, returning one TurnResponse per assistant turn."""
    if not specs:
        return []
    messages = [_init_messages(s) for s in specs]
    max_turns = max(len(s.followups) for s in specs) + 1
    results: list[TurnResponse] = []

    for t in range(max_turns):
        active = [i for i, s in enumerate(specs) if t <= len(s.followups)]
        if not active:
            break
        batch = [messages[i] for i in active]
        outs = client.generate(batch, n=1, temperature=temperature,
                                max_new_tokens=max_new_tokens)
        for k, i in enumerate(active):
            resp = outs[k][0]
            messages[i].append(Message("assistant", resp))
            results.append(TurnResponse(
                conv_id=specs[i].conv_id, category=specs[i].category,
                turn=t + 1, response=resp, meta=specs[i].meta))
            if t < len(specs[i].followups):
                fu = specs[i].followups[t]
                if followup_suffix:
                    fu = f"{fu} {followup_suffix}"
                messages[i].append(Message("user", fu))
    return results
