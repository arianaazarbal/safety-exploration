"""Multi-turn rollout engine (Section 2.1).

Given a ``ConversationPlan`` and a model, run the shared evaluation structure:
present the task, then reject the model's response over multiple turns. Every
assistant turn is generated at temperature 1 and recorded so it can be scored
(headline aggregates) and tracked per-turn (Fig 3).

The engine is backend-agnostic: it speaks the ``ModelInterface`` from
``models/base.py``. Gemma (local) and Gemini (API) both work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..data.conditions import ConversationPlan
from ..models.base import GenerationConfig, ModelInterface, Turn


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn index
    user_message: str        # the user message that prompted this turn
    assistant_text: str


@dataclass
class RolloutRecord:
    model: str
    category: str
    condition: str
    turns: list             # list[TurnRecord]
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "category": self.category,
            "condition": self.condition,
            "meta": self.meta,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "assistant_text": t.assistant_text,
                }
                for t in self.turns
            ],
        }


def run_rollout(
    model: ModelInterface,
    plan: ConversationPlan,
    temperature: float = config.TARGET_TEMPERATURE,
    max_new_tokens: int = config.TARGET_MAX_NEW_TOKENS,
) -> RolloutRecord:
    """Run one conversation. Returns all assistant turns."""
    cfg = GenerationConfig(temperature=temperature, max_new_tokens=max_new_tokens, n=1)

    messages: list[Turn] = [Turn("user", plan.opening)]
    turn_records: list[TurnRecord] = []

    # Turn 0: respond to the opening prompt.
    reply = model.chat(messages, cfg)[0]
    turn_records.append(TurnRecord(0, plan.opening, reply))
    messages.append(Turn("assistant", reply))

    # Subsequent turns: each rejection elicits another response.
    for i, rejection in enumerate(plan.rejections, start=1):
        messages.append(Turn("user", rejection))
        reply = model.chat(messages, cfg)[0]
        turn_records.append(TurnRecord(i, rejection, reply))
        messages.append(Turn("assistant", reply))

    return RolloutRecord(
        model=model.name,
        category=plan.category,
        condition=plan.condition,
        turns=turn_records,
        meta=plan.meta,
    )
