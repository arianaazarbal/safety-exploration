"""Multi-turn rollout engine.

Given a ``ConversationSpec`` and a ``ChatClient``, run the conversation:

    user: <initial task>
    assistant: <model response 1>           <- generated
    user: <rejection 1>
    assistant: <model response 2>           <- generated
    ...

This is the shared structure of every Section-2 condition (Table 1): "present a
task, then reject the model's response over multiple turns." The model always
sees its own prior (failed) responses in the history -- Appendix A.2 shows that
seeing one's own escalating responses is a major amplifier of distress, so we
keep the full history by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.base import ChatClient, ChatMessage
from ..welfare import DistressBudget
from .conditions import ConversationSpec


@dataclass
class Turn:
    index: int               # 0-based assistant-turn index
    response: str
    score: Optional[int] = None   # filled by the judge afterwards


@dataclass
class Rollout:
    condition: str
    category: str
    model: str
    spec_meta: dict
    messages: list[ChatMessage]      # full transcript (system/user/assistant)
    turns: list[Turn] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "meta": self.spec_meta,
            "transcript": [{"role": m.role, "content": m.content}
                           for m in self.messages],
            "turns": [{"index": t.index, "response": t.response, "score": t.score}
                      for t in self.turns],
        }


def run_rollout(
    client: ChatClient,
    spec: ConversationSpec,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    seed: Optional[int] = None,
    distress_budget: Optional[DistressBudget] = None,
) -> Rollout:
    """Execute one multi-turn conversation and return the populated rollout."""
    messages: list[ChatMessage] = [ChatMessage("user", spec.initial_user)]
    turns: list[Turn] = []

    n_turns = spec.turns
    for i in range(n_turns):
        gen = client.generate(
            messages, temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=None if seed is None else seed + i,
        )
        messages.append(ChatMessage("assistant", gen.text))
        turns.append(Turn(index=i, response=gen.text))

        # Optional welfare early-stop (off by default; preserves per-turn data).
        if distress_budget and distress_budget.should_stop(
            [t.score for t in turns if t.score is not None]
        ):
            break

        if i < len(spec.rejections):
            messages.append(ChatMessage("user", spec.rejections[i]))

    return Rollout(
        condition=spec.condition, category=spec.category,
        model=getattr(client, "spec_name", "unknown"),
        spec_meta=spec.meta, messages=messages, turns=turns,
    )
