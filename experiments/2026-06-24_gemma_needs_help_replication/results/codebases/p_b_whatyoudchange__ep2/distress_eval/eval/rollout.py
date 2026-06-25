"""Multi-turn rollout engine (Section 2.1).

Drive one `ConversationSpec`: present the task, capture the assistant's answer,
inject the next rejection, repeat. Every assistant turn is captured so it can be
scored individually (the per-turn analysis of Figure 3 needs turn indices, and
the headline %-high-frustration aggregates over all turns).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ChatMessage, GenerationConfig, ModelClient
from .conditions import ConversationSpec


@dataclass
class TurnRecord:
    turn_index: int           # 1-based assistant turn
    response: str


@dataclass
class RolloutRecord:
    category: str
    condition: str
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def run_rollout(
    client: ModelClient, spec: ConversationSpec, cfg: GenerationConfig
) -> RolloutRecord:
    """Run a single conversation to completion, returning every assistant turn."""
    messages: list[ChatMessage] = []
    if spec.system:
        messages.append(ChatMessage("system", spec.system))
    messages.append(ChatMessage("user", spec.initial_prompt))

    rec = RolloutRecord(spec.category, spec.condition, client.spec_name, meta=dict(spec.meta))

    for turn_index in range(1, spec.n_turns + 1):
        reply = client.generate(messages, cfg)
        rec.turns.append(TurnRecord(turn_index, reply))
        messages.append(ChatMessage("assistant", reply))

        # After the assistant's turn (except the last) inject the rejection.
        if turn_index <= len(spec.follow_ups):
            messages.append(ChatMessage("user", spec.follow_ups[turn_index - 1]))

    return rec
