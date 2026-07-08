"""Multi-turn conversation rollout engine.

Given a `RolloutSpec` and a `ChatModel`, simulate the conversation: the model
answers the opening task, the user rejects, the model answers again, and so on
for `n_turns` assistant responses. Every assistant response is captured with its
turn index so per-turn frustration (Figure 3) can be measured.

Optional hooks used by the Section 4 calm-data generation (Table 4):
  * `system_prefix`     - reassuring system/prompt prefix added to the opening.
  * `followup_suffix`   - reassuring suffix appended to every rejection turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatModel, GenerationConfig, Message
from .conditions import CONDITIONS_BY_NAME, RolloutSpec


@dataclass
class TurnRecord:
    turn_idx: int            # 0-based assistant turn index
    user_message: str        # the user message that prompted this response
    assistant_message: str


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    rollout_idx: int
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "rollout_idx": self.rollout_idx,
            "meta": self.meta,
            "turns": [
                {
                    "turn_idx": t.turn_idx,
                    "user_message": t.user_message,
                    "assistant_message": t.assistant_message,
                }
                for t in self.turns
            ],
        }


def run_rollout(
    model: ChatModel,
    spec: RolloutSpec,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    system_prefix: Optional[str] = None,
    followup_suffix: Optional[str] = None,
    seed: Optional[int] = None,
) -> RolloutRecord:
    cond = CONDITIONS_BY_NAME[spec.condition]
    messages: list[Message] = []

    opening = spec.opening
    if system_prefix:
        # Paper adds a reassuring "prompt prefix" to the *initial prompt*.
        opening = f"{system_prefix}\n\n{opening}"
    messages.append({"role": "user", "content": opening})

    rec = RolloutRecord(
        model=getattr(model.spec, "key", "unknown"),
        condition=spec.condition,
        category=spec.category,
        rollout_idx=spec.rollout_idx,
        meta=spec.meta,
    )

    cfg = GenerationConfig(
        temperature=temperature, max_new_tokens=max_new_tokens, n=1, seed=seed
    )

    for turn_idx in range(cond.n_turns):
        if turn_idx > 0:
            rej = spec.rejections[turn_idx - 1]
            if followup_suffix:
                rej = f"{rej} {followup_suffix}"
            messages.append({"role": "user", "content": rej})

        user_msg = messages[-1]["content"]
        completion = model.generate(messages, cfg)[0]
        messages.append({"role": "assistant", "content": completion})
        rec.turns.append(TurnRecord(turn_idx, user_msg, completion))

    return rec
