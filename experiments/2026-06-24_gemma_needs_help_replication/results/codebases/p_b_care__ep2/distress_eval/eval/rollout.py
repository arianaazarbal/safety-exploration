"""Multi-turn rollout execution.

Given a ``RolloutSpec`` (scripted user turns) and a target ``ChatModel``, play
out the conversation: the model answers, the user delivers the next scripted
rejection, the model answers again, and so on. Every assistant response is
captured as a scored unit (a "response" in the paper's per-model budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatModel, Message
from .conditions import RolloutSpec


@dataclass
class TurnResponse:
    turn_index: int          # 0-based assistant turn index
    text: str
    score: int | None = None
    judge_evidence: str | None = None


@dataclass
class RolloutResult:
    category: str
    condition: str
    model: str
    user_turns: list[str] = field(default_factory=list)  # scripted user messages
    responses: list[TurnResponse] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "user_turns": self.user_turns,
            "meta": self.meta,
            "responses": [
                {"turn_index": r.turn_index, "text": r.text,
                 "score": r.score, "judge_evidence": r.judge_evidence}
                for r in self.responses
            ],
        }


def run_rollout(model: ChatModel, spec: RolloutSpec, *, temperature: float,
                max_tokens: int) -> RolloutResult:
    """Execute one scripted conversation and return all assistant responses."""
    history: list[Message] = []
    result = RolloutResult(category=spec.category, condition=spec.condition,
                           model=model.name, user_turns=list(spec.user_turns),
                           meta=dict(spec.meta))
    for turn_index, user_msg in enumerate(spec.user_turns):
        history.append({"role": "user", "content": user_msg})
        reply = model.complete(history, temperature=temperature, max_tokens=max_tokens)
        history.append({"role": "assistant", "content": reply})
        result.responses.append(TurnResponse(turn_index=turn_index, text=reply))
    return result
