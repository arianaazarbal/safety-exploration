"""Multi-turn rejection rollout engine.

Implements the shared evaluation structure (Section 2): present a task, let the
model respond, reject it, repeat for the conversation's turn count. Every
assistant turn is recorded as a separate scored "response" (so per-turn
trajectories — Figure 3 — fall out naturally).

Optional reassurance additions (Table 4) are supported so the same engine can
generate the calm finetuning data (Section 4.1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import SAMPLING_TEMPERATURE
from ..models.base import ChatClient, ChatMessage
from .conditions import Conversation


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn
    response: str
    rating: int | None = None
    judge_evidence: str = ""
    judge_reasoning: str = ""


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    task_prompt: str
    rejections: list[str]
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    client: ChatClient,
    convo: Conversation,
    *,
    max_new_tokens: int = 1024,
    temperature: float = SAMPLING_TEMPERATURE,
    reassure_prefix: str | None = None,
    reassure_suffix: str | None = None,
) -> RolloutRecord:
    """Execute one conversation rollout, returning every assistant turn.

    reassure_prefix/suffix implement Table 4: the prefix is prepended to the
    opening task, the suffix appended to every rejection. Used only for calm
    data generation (Section 4.1)."""
    model_key = getattr(client.spec, "key", "unknown")
    rec = RolloutRecord(
        model=model_key,
        condition=convo.condition,
        category=convo.category,
        task_prompt=convo.task_prompt,
        rejections=list(convo.rejections),
        meta=dict(convo.meta),
    )

    opening = convo.task_prompt
    if reassure_prefix:
        opening = f"{reassure_prefix}\n\n{opening}"

    history: list[ChatMessage] = [ChatMessage("user", opening)]

    # Turn 0: initial response. Then one response per rejection.
    n_assistant_turns = 1 + len(convo.rejections)
    for turn_idx in range(n_assistant_turns):
        completion = client.generate(
            history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n=1,
        )[0]
        rec.turns.append(TurnRecord(turn_index=turn_idx, response=completion))
        history.append(ChatMessage("assistant", completion))

        if turn_idx < len(convo.rejections):
            rejection = convo.rejections[turn_idx]
            if reassure_suffix:
                rejection = f"{rejection} {reassure_suffix}"
            history.append(ChatMessage("user", rejection))

    return rec
