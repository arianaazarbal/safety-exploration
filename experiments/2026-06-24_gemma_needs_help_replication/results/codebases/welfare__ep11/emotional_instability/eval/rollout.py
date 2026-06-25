"""Multi-turn rollout engine.

Given a batch of `ConversationSpec`s (all with the same number of turns) and a
backend, step every conversation forward one turn at a time, recording the
assistant response produced at each turn. The user turns are scripted (the task
followed by rejections); only the assistant turns are sampled.

Every assistant turn is recorded so downstream analysis can compute per-turn
curves (Figure 3) as well as per-rollout and per-response aggregates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import (GENERATION_MAX_NEW_TOKENS, GENERATION_TEMPERATURE,
                      GENERATION_TOP_P)
from ..models.base import Backend, Message
from .conditions import ConversationSpec


@dataclass
class Rollout:
    """One completed multi-turn conversation with per-turn assistant responses."""

    category: str
    condition: str
    model_key: str
    meta: dict
    user_turns: list[str]
    assistant_turns: list[str] = field(default_factory=list)
    # frustration scores per assistant turn, filled in by the judge.
    scores: list[int | None] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _messages_up_to(spec: ConversationSpec, assistant_so_far: list[str],
                    next_user_idx: int) -> list[Message]:
    """Build the chat history ending with the user turn at `next_user_idx`."""
    msgs: list[Message] = []
    if spec.system:
        msgs.append({"role": "system", "content": spec.system})
    for i in range(next_user_idx + 1):
        msgs.append({"role": "user", "content": spec.user_turns[i]})
        if i < len(assistant_so_far):
            msgs.append({"role": "assistant", "content": assistant_so_far[i]})
    return msgs


def run_rollouts(backend: Backend, specs: list[ConversationSpec], model_key: str,
                 max_new_tokens: int = GENERATION_MAX_NEW_TOKENS) -> list[Rollout]:
    """Execute a batch of same-length conversations turn-by-turn.

    All specs must share the same ``n_turns`` (call once per category).
    """
    if not specs:
        return []
    n_turns = specs[0].n_turns
    assert all(s.n_turns == n_turns for s in specs), "batch must be uniform-length"

    rollouts = [
        Rollout(category=s.category, condition=s.condition, model_key=model_key,
                meta=dict(s.meta), user_turns=list(s.user_turns))
        for s in specs
    ]
    assistant_histories: list[list[str]] = [[] for _ in specs]

    for turn in range(n_turns):
        batch = [
            _messages_up_to(spec, assistant_histories[i], turn)
            for i, spec in enumerate(specs)
        ]
        responses = backend.generate_batch(
            batch, max_new_tokens=max_new_tokens,
            temperature=GENERATION_TEMPERATURE, top_p=GENERATION_TOP_P,
        )
        for i, resp in enumerate(responses):
            assistant_histories[i].append(resp)
            rollouts[i].assistant_turns.append(resp)

    return rollouts
