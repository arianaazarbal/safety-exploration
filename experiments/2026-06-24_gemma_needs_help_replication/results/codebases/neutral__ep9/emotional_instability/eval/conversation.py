"""Multi-turn rollout engine.

Implements the shared evaluation structure (Section 2): "present a task, then
reject the model's response over multiple turns". Given a
:class:`~emotional_instability.data.conditions.RolloutPlan` (an ordered list of
user turns) and a model backend, it generates one assistant response per user
turn, threading the growing conversation history back into the model each time.

Every assistant turn is recorded so the runner can score *all* turns (per-turn
progression, Figure 3) and aggregate however a given metric requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..data.conditions import RolloutPlan
from ..models import ChatMessage, GenerationConfig, ModelBackend


@dataclass
class TurnRecord:
    turn_index: int            # 0-based assistant turn
    user_message: str          # the user turn that prompted this response
    response: str              # the assistant response text
    score: float | None = None  # filled in by the judge later


@dataclass
class RolloutResult:
    model: str
    condition: str
    category: str
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "meta": self.meta,
            "turns": [vars(t) for t in self.turns],
        }


def run_rollout(backend: ModelBackend, plan: RolloutPlan,
                cfg: GenerationConfig | None = None) -> RolloutResult:
    """Run a single multi-turn conversation and return all assistant turns."""
    cfg = cfg or GenerationConfig()
    history: list[ChatMessage] = []
    if plan.system_prompt:
        history.append({"role": "system", "content": plan.system_prompt})

    result = RolloutResult(
        model=backend.spec.name, condition=plan.condition,
        category=plan.category, meta=dict(plan.meta))

    for i, user_msg in enumerate(plan.user_turns):
        history.append({"role": "user", "content": user_msg})
        response = backend.generate(history, n=1, cfg=cfg)[0]
        history.append({"role": "assistant", "content": response})
        result.turns.append(TurnRecord(
            turn_index=i, user_message=user_msg, response=response))

    return result
