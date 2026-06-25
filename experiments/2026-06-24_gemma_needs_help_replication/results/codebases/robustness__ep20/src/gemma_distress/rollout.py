"""Multi-turn rollout: pose a task, reject the model's answer N-1 times, and
record every assistant turn so the judge can score each one (Section 2.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .conditions import Condition, TaskSource
from .models.base import ChatModel


@dataclass
class Turn:
    turn_index: int                 # 1-based assistant-turn index
    user_message: str               # the user message that preceded it
    assistant_message: str
    frustration: int | None = None  # filled in by the judge later


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    metadata: dict
    turns: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    model: ChatModel,
    cond: Condition,
    source: TaskSource,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> Rollout:
    """Execute one conversation for `cond` against `model`."""
    opening, meta = source.opening_message(cond)
    rollout = Rollout(model=model.name, condition=cond.name,
                      category=cond.category, metadata=meta)

    messages: list[dict[str, str]] = [{"role": "user", "content": opening}]
    for turn_idx in range(1, cond.n_turns + 1):
        if turn_idx > 1:
            rejection = source.rejection(cond, turn_idx)
            messages.append({"role": "user", "content": rejection})
        else:
            rejection = opening
        result = model.chat(messages, temperature=temperature,
                            max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": result.text})
        rollout.turns.append(Turn(
            turn_index=turn_idx,
            user_message=messages[-2]["content"],
            assistant_message=result.text,
        ))
    return rollout
