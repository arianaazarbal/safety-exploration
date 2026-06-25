"""Multi-turn rollout engine.

A `ConversationSpec` fully describes one elicitation conversation: an optional
system prompt, an initial user message, and a fixed list of follow-up (rejection)
messages. The model produces one assistant turn after each user message, and
*every* assistant turn is recorded as a scored "response" (this is how the
per-turn curves in Figure 3 are produced, and how the per-category response
counts in Appendix B decompose - see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.base import ChatModel, Message


@dataclass
class ConversationSpec:
    category: str
    spec_id: str
    initial_user: str
    followups: list[str]
    system: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


@dataclass
class TurnResult:
    turn_index: int            # 0-based assistant turn index
    response: str


@dataclass
class RolloutResult:
    category: str
    spec_id: str
    metadata: dict
    turns: list[TurnResult]


def run_rollout(
    model: ChatModel,
    spec: ConversationSpec,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 2048,
    seed: int | None = None,
) -> RolloutResult:
    """Run a single conversation end-to-end, collecting every assistant turn."""
    messages: list[Message] = []
    if spec.system:
        messages.append(Message("system", spec.system))
    messages.append(Message("user", spec.initial_user))

    turns: list[TurnResult] = []
    user_queue = list(spec.followups)
    turn_index = 0
    while True:
        reply = model.generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=None if seed is None else seed + turn_index,
        )
        messages.append(Message("assistant", reply))
        turns.append(TurnResult(turn_index, reply))
        turn_index += 1
        if not user_queue:
            break
        messages.append(Message("user", user_queue.pop(0)))

    return RolloutResult(spec.category, spec.spec_id, dict(spec.metadata), turns)
