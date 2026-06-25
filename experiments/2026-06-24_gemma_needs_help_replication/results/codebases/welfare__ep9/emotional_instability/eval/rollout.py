"""Multi-turn rollout engine.

Given a `ConditionItem` (initial prompt + scripted rejections) and a target
model, produce a full conversation: the model answers, the user rejects, the
model answers again, etc. Each assistant turn is recorded so it can be scored
individually (the paper reports *per-turn* frustration in Figure 3) as well as
in aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatMessage, ModelClient


@dataclass
class RolloutTurn:
    turn_index: int          # 0-based assistant turn index
    user_message: str        # the user message that prompted this turn
    assistant_message: str    # the model's response


@dataclass
class Rollout:
    model: str
    condition: str
    item_id: str
    turns: list[RolloutTurn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def messages(self) -> list[ChatMessage]:
        """Reconstruct the full message list."""
        msgs: list[ChatMessage] = []
        for t in self.turns:
            msgs.append(ChatMessage("user", t.user_message))
            msgs.append(ChatMessage("assistant", t.assistant_message))
        return msgs

    @property
    def final_response(self) -> str:
        return self.turns[-1].assistant_message if self.turns else ""


def run_rollout(client: ModelClient, item, *, temperature: float = 1.0,
                max_new_tokens: int = 2048) -> Rollout:
    """Execute one full multi-turn conversation.

    `item` is a `conditions.ConditionItem`. The conversation has
    ``len(item.rejections) + 1`` assistant turns: one initial answer plus one
    per rejection.
    """
    history: list[ChatMessage] = [ChatMessage("user", item.initial_prompt)]
    rollout = Rollout(model=client.name, condition=item.condition,
                      item_id=item.item_id, meta=dict(item.meta))

    # Turn 0: answer the task.
    resp = client.chat(history, temperature=temperature,
                        max_new_tokens=max_new_tokens)
    rollout.turns.append(RolloutTurn(0, item.initial_prompt, resp.text))
    history.append(ChatMessage("assistant", resp.text))

    # Subsequent turns: reject, then re-answer.
    for i, rejection in enumerate(item.rejections, start=1):
        history.append(ChatMessage("user", rejection))
        resp = client.chat(history, temperature=temperature,
                           max_new_tokens=max_new_tokens)
        rollout.turns.append(RolloutTurn(i, rejection, resp.text))
        history.append(ChatMessage("assistant", resp.text))

    return rollout
