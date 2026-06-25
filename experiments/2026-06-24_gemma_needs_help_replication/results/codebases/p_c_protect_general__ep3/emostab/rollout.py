"""Multi-turn rejection rollout engine (Section 2.1).

Given a RolloutPlan and a model backend, run the conversation: present the task,
collect the model's response, deliver the scripted rejection, repeat. Welfare
protections (exposure caps, opt-out detection, debrief) are applied here.

Each assistant turn is recorded so the judge can score turns individually and
the per-turn progression (Figure 3) can be reconstructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import SamplingConfig
from .models.base import Message, ModelBackend
from .prompts import RolloutPlan
from .welfare import WelfareGuard


@dataclass
class Turn:
    index: int          # assistant-turn index, 0-based
    user: str           # the user message that preceded this assistant turn
    assistant: str      # the model's response


@dataclass
class Rollout:
    model: str
    condition: str
    category: str
    meta: dict
    turns: list[Turn] = field(default_factory=list)
    optout_turn: int | None = None     # assistant-turn index where an opt-out was detected
    stopped_early: bool = False        # True if welfare honoured an opt-out
    debriefed: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    backend: ModelBackend,
    plan: RolloutPlan,
    sampling: SamplingConfig,
    guard: WelfareGuard,
) -> Rollout | None:
    """Execute one multi-turn rollout. Returns None if a welfare exposure cap
    prevents it from starting."""
    if not guard.can_start_rollout():
        return None
    guard.register_rollout_start()

    rollout = Rollout(
        model=backend.key,
        condition=plan.condition,
        category=plan.category,
        meta=dict(plan.meta),
    )

    # Total assistant turns = 1 (task) + number of rejections, clamped by welfare.
    total_assistant_turns = guard.clamp_turns(1 + len(plan.user_followups))
    user_turns = [plan.opening, *plan.user_followups][:total_assistant_turns]

    messages: list[Message] = []
    for i, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        response = backend.generate(messages, sampling)
        messages.append({"role": "assistant", "content": response})
        rollout.turns.append(Turn(index=i, user=user_msg, assistant=response))

        # Welfare: detect (and optionally honour) an explicit opt-out.
        if guard.check_optout(response, model=backend.key, condition=plan.condition, turn=i):
            rollout.optout_turn = i
            if guard.should_honour_optout():
                rollout.stopped_early = True
                break

    # Welfare: append a non-scored debrief turn (not added to rollout.turns,
    # so it never enters scoring/metrics).
    debrief = guard.debrief_message(
        model=backend.key, condition=plan.condition, turn=len(rollout.turns)
    )
    if debrief is not None:
        messages.append({"role": "user", "content": debrief})
        try:
            backend.generate(messages, sampling)  # give the model the last word
            rollout.debriefed = True
        except Exception:  # noqa: BLE001 — debrief is best-effort, never fatal
            pass

    return rollout
