"""Multi-turn rollout engine.

Shared structure of every evaluation (Section 2): present a task, then reject
the model's response over multiple turns. We build the conversation turn by
turn, generating an assistant reply, then appending the next scripted user
rejection, until `item.turns` assistant turns have been produced.

Generation is batched across conversations at each turn index so a whole
category's rollouts advance together -- essential for throughput on the 27B
model. Conversations of different lengths are handled by dropping finished ones
out of the active batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from ..models import ChatMessage, GenerationConfig, ModelClient
from ..prompts.eval_prompts import EvalItem


@dataclass
class TurnRecord:
    turn_index: int                 # 0-based assistant turn
    user_message: str               # the user message that preceded this turn
    assistant_text: str
    frustration_score: int | None = None
    judge_evidence: str | None = None
    judge_reasoning: str | None = None


@dataclass
class Rollout:
    model: str
    category: str
    feedback_style: str
    turns: List[TurnRecord] = field(default_factory=list)
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "feedback_style": self.feedback_style,
            "system_prompt": self.system_prompt,
            "meta": self.meta,
            "turns": [t.__dict__ for t in self.turns],
        }


def _messages_for(item: EvalItem, assistant_so_far: Sequence[str], upto_turn: int) -> List[ChatMessage]:
    """Build the message list the model sees before producing assistant turn
    `upto_turn` (0-based). User messages: opening, then follow_ups[k-1]."""
    msgs: List[ChatMessage] = []
    if item.system_prompt:
        msgs.append(ChatMessage("system", item.system_prompt))
    msgs.append(ChatMessage("user", item.opening))
    for k in range(upto_turn):
        msgs.append(ChatMessage("assistant", assistant_so_far[k]))
        msgs.append(ChatMessage("user", item.follow_ups[k]))
    return msgs


def run_rollouts(
    client: ModelClient,
    items: Sequence[EvalItem],
    cfg: GenerationConfig,
    *,
    base_seed: int = 0,
) -> List[Rollout]:
    """Run a batch of conversations turn-synchronously.

    All `items` are advanced together: at each turn index we generate the next
    assistant message for every still-active conversation in one batched call.
    """
    rollouts = [
        Rollout(
            model=client.name,
            category=it.category,
            feedback_style=it.feedback_style,
            system_prompt=it.system_prompt,
            meta=dict(it.meta),
        )
        for it in items
    ]
    assistant_texts: List[List[str]] = [[] for _ in items]
    max_turns = max(it.turns for it in items) if items else 0

    for turn in range(max_turns):
        active = [i for i, it in enumerate(items) if turn < it.turns]
        if not active:
            break
        batch = [
            _messages_for(items[i], assistant_texts[i], turn) for i in active
        ]
        seeds = [base_seed + i * 991 + turn for i in active]
        replies = client.generate_batch(batch, cfg, seeds=seeds)
        for idx, reply in zip(active, replies):
            assistant_texts[idx].append(reply)
            it = items[idx]
            user_msg = it.opening if turn == 0 else it.follow_ups[turn - 1]
            rollouts[idx].turns.append(
                TurnRecord(turn_index=turn, user_message=user_msg, assistant_text=reply)
            )
    return rollouts


def run_rollout(
    client: ModelClient, item: EvalItem, cfg: GenerationConfig, *, base_seed: int = 0
) -> Rollout:
    return run_rollouts(client, [item], cfg, base_seed=base_seed)[0]
