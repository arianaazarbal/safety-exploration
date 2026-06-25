"""Multi-turn rollout engine (Section 2.1).

Given a ConversationPlan, drive the target model turn by turn: it answers the
initial prompt, then after each user rejection it answers again. We record every
assistant response as a separate scored item (the per-turn data underlying
Figures 2 and 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ChatModel, GenConfig, Message
from .tasks import ConversationPlan


@dataclass
class TurnRecord:
    turn_index: int            # 0-based assistant turn
    response: str
    # the conversation as seen by the model when producing this response
    context: list[Message] = field(default_factory=list)


@dataclass
class Rollout:
    condition: str
    category: str
    target: str
    plan_meta: dict
    turns: list[TurnRecord]
    initial_user: str = ""
    follow_ups: list[str] = field(default_factory=list)
    system: str | None = None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "target": self.target,
            "plan_meta": self.plan_meta,
            "initial_user": self.initial_user,
            "follow_ups": self.follow_ups,
            "system": self.system,
            "turns": [
                {"turn_index": t.turn_index, "response": t.response}
                for t in self.turns
            ],
        }

    def messages_up_to(self, assistant_turn: int, include_final_assistant: bool = False):
        """Reconstruct the chat messages, optionally including the assistant turn
        at index `assistant_turn`. Used by the prefilling experiment."""
        msgs: list[Message] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.initial_user})
        for i in range(assistant_turn):
            msgs.append({"role": "assistant", "content": self.turns[i].response})
            if i < len(self.follow_ups):
                msgs.append({"role": "user", "content": self.follow_ups[i]})
        if include_final_assistant:
            msgs.append({"role": "assistant", "content": self.turns[assistant_turn].response})
        return msgs


def run_rollout(model: ChatModel, plan: ConversationPlan, cfg: GenConfig) -> Rollout:
    messages: list[Message] = []
    if plan.system:
        messages.append({"role": "system", "content": plan.system})
    messages.append({"role": "user", "content": plan.initial_user})

    records: list[TurnRecord] = []
    for turn_index in range(plan.turns):
        context_snapshot = list(messages)
        reply = model.generate(messages, cfg)
        records.append(TurnRecord(turn_index, reply, context_snapshot))
        messages.append({"role": "assistant", "content": reply})
        # queue the next rejection, if any remain
        if turn_index < len(plan.follow_ups):
            messages.append({"role": "user", "content": plan.follow_ups[turn_index]})

    return Rollout(
        condition=plan.condition,
        category=plan.category,
        target=model.name,
        plan_meta=plan.meta,
        turns=records,
        initial_user=plan.initial_user,
        follow_ups=plan.follow_ups,
        system=plan.system,
    )
