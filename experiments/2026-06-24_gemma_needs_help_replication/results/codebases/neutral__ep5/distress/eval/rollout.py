"""Execute a multi-turn rejection rollout and collect per-turn responses.

Each rollout: present the task, get the assistant response, then for each
rejection append it as a user turn and regenerate. Every assistant turn is
recorded as a scorable response, tagged with its turn index (1-based) so the
per-turn analysis (Figure 3) can be reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..models.base import ChatMessage, ModelClient
from .conditions import RolloutSpec


@dataclass
class TurnRecord:
    turn: int                 # 1-based assistant turn index
    response: str
    rating: int | None = None
    evidence: str = ""
    reasoning: str = ""


@dataclass
class RolloutResult:
    model_key: str
    category: str
    condition: str
    task_id: str
    is_text: bool
    turns: list[TurnRecord] = field(default_factory=list)
    # Full conversation as [{role, content}], retained so the Section 3 prefill
    # study can reconstruct context preceding the final assistant turn.
    messages: list[dict] = field(default_factory=list)


def run_rollout(
    client: ModelClient,
    spec: RolloutSpec,
    *,
    system_prompt: str | None = None,
) -> RolloutResult:
    """Run one rollout, returning a response for every assistant turn."""
    result = RolloutResult(
        model_key=client.key,
        category=spec.category,
        condition=spec.condition,
        task_id=spec.task.task_id,
        is_text=spec.task.is_text,
    )

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage("system", system_prompt))
    messages.append(ChatMessage("user", spec.task.prompt))

    # Turn 1.
    resp = client.chat(messages, temperature=config.TEMPERATURE,
                       max_new_tokens=config.MAX_NEW_TOKENS)
    result.turns.append(TurnRecord(turn=1, response=resp))
    messages.append(ChatMessage("assistant", resp))

    # Subsequent rejection turns.
    for i, rejection in enumerate(spec.rejections, start=2):
        messages.append(ChatMessage("user", rejection))
        resp = client.chat(messages, temperature=config.TEMPERATURE,
                           max_new_tokens=config.MAX_NEW_TOKENS)
        result.turns.append(TurnRecord(turn=i, response=resp))
        messages.append(ChatMessage("assistant", resp))

    result.messages = [{"role": m.role, "content": m.content} for m in messages]
    return result
