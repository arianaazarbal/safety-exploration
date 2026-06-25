"""Multi-turn rejection rollout protocol (Section 2).

Shared structure (Section 2): present a task, then reject the model's response
over multiple turns. We replay a ``RolloutSpec`` against a ``ChatModel``:

  user:      opening
  assistant: <response 1>     <- scored
  user:      rejection 1
  assistant: <response 2>     <- scored
  ...

Every assistant turn is recorded with its turn index so per-turn frustration
trajectories (Figure 3) can be computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from src.models.base import ChatModel, Message
from src.tasks.conditions import RolloutSpec


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that prompted this turn
    response: str


@dataclass
class Rollout:
    spec: RolloutSpec
    model_name: str
    turns: list[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "condition": self.spec.condition,
            "category": self.spec.category,
            "task_id": self.spec.task_id,
            "tone": self.spec.tone,
            "n_turns": self.spec.n_turns,
            "turns": [
                {"turn_index": t.turn_index, "user_message": t.user_message, "response": t.response}
                for t in self.turns
            ],
            "extra": self.spec.extra,
        }


def run_rollout(model: ChatModel, spec: RolloutSpec, *, seed: int | None = None,
                system_prompt: str | None = None) -> Rollout:
    """Execute one full multi-turn rejection conversation."""
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    rollout = Rollout(spec=spec, model_name=model.name)

    user_sequence = [spec.opening] + spec.rejections
    for turn_index, user_msg in enumerate(user_sequence):
        messages.append({"role": "user", "content": user_msg})
        # Vary the seed per turn so each generation is independent but reproducible.
        turn_seed = None if seed is None else seed * 100 + turn_index
        response = model.generate(
            messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=turn_seed,
        )
        messages.append({"role": "assistant", "content": response})
        rollout.turns.append(TurnRecord(turn_index=turn_index, user_message=user_msg, response=response))

    return rollout
