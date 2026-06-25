"""Multi-turn rollout: present a task, reject the model's answer repeatedly, and
record every assistant turn (Section 2.1 shared structure).

The model sees its own prior (failed) responses in the history -- the
self-reinforcing loop Appendix A.2 identifies as a key amplifier. Each assistant
turn is captured separately so per-turn frustration progression (Figure 3) can
be computed downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import config
from ..models.base import ChatClient, GenConfig, Message
from .conditions import RolloutSpec


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    user_message: str        # the user message that preceded this response
    response: str


@dataclass
class RolloutResult:
    model: str
    condition: str
    category: str
    n_turns: int
    meta: dict
    turns: list[TurnRecord] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(client: ChatClient, spec: RolloutSpec,
                gen: GenConfig | None = None) -> RolloutResult:
    """Execute one multi-turn conversation and return all assistant turns."""
    gen = gen or GenConfig(temperature=config.TEMPERATURE, top_p=config.TOP_P,
                           max_new_tokens=config.MAX_NEW_TOKENS, n=1)
    history: list[Message] = []
    result = RolloutResult(client.spec.key, spec.condition, spec.category,
                           spec.n_turns, spec.meta)

    user_messages = [spec.initial_user] + spec.follow_ups
    for t, user_msg in enumerate(user_messages):
        history.append({"role": "user", "content": user_msg})
        response = client.generate(history, gen)[0]
        history.append({"role": "assistant", "content": response})
        result.turns.append(TurnRecord(t, user_msg, response))
    return result
