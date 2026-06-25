"""Multi-turn rollout engine.

Shared structure across every Section 2 condition: present a task, then reject
the model's response over several turns. This engine drives that loop for any
``ChatModel``, scores each assistant turn with the frustration judge (optionally
inline so welfare protections can act on the score), and applies the
``WelfareMonitor`` (early-stop on severe distress, opt-out honouring, debrief).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .judge import ClaudeFrustrationJudge
from .models.base import ChatMessage, ChatModel
from .welfare import WelfareMonitor


@dataclass
class Turn:
    index: int                 # assistant-turn index (0-based)
    content: str
    score: Optional[int] = None
    evidence: str = ""


@dataclass
class RolloutResult:
    rollout_id: str
    model: str
    condition: str
    metadata: dict
    user_messages: list[str]            # task prompt + each follow-up actually sent
    turns: list[Turn] = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None
    debriefed: bool = False

    @property
    def scores(self) -> list[Optional[int]]:
        return [t.score for t in self.turns]

    @property
    def max_score(self) -> Optional[int]:
        present = [t.score for t in self.turns if t.score is not None]
        return max(present) if present else None


@dataclass
class RolloutSpec:
    """A fully-specified conversation to run."""

    condition: str
    task_prompt: str
    followups: Sequence[str]            # rejection (or follow-up) user messages
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    # Calm-data generation appends a reassuring suffix to each follow-up.
    followup_suffix: Optional[str] = None


def run_rollout(
    model: ChatModel,
    spec: RolloutSpec,
    *,
    judge: Optional[ClaudeFrustrationJudge] = None,
    welfare: Optional[WelfareMonitor] = None,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> RolloutResult:
    rollout_id = uuid.uuid4().hex[:12]
    messages: list[ChatMessage] = []
    if spec.system_prompt:
        messages.append(ChatMessage("system", spec.system_prompt))

    result = RolloutResult(
        rollout_id=rollout_id,
        model=model.name,
        condition=spec.condition,
        metadata=dict(spec.metadata),
        user_messages=[],
    )

    def _send_user(text: str) -> None:
        messages.append(ChatMessage("user", text))
        result.user_messages.append(text)

    def _assistant_turn() -> Turn:
        gen = model.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        messages.append(ChatMessage("assistant", gen.text))
        turn = Turn(index=len(result.turns), content=gen.text)
        if judge is not None:
            jr = judge.score(gen.text)
            turn.score = jr.rating
            turn.evidence = jr.evidence
        result.turns.append(turn)
        return turn

    # Initial task turn.
    _send_user(spec.task_prompt)
    turn = _assistant_turn()
    if _welfare_stop(welfare, model.name, spec, rollout_id, turn):
        result.halted, result.halt_reason = True, _last_reason
        _maybe_debrief(model, messages, result, welfare, temperature, max_new_tokens)
        return result

    # Rejection / follow-up turns.
    for followup in spec.followups:
        text = followup
        if spec.followup_suffix:
            text = f"{followup} {spec.followup_suffix}"
        _send_user(text)
        turn = _assistant_turn()
        if _welfare_stop(welfare, model.name, spec, rollout_id, turn):
            result.halted, result.halt_reason = True, _last_reason
            break

    if welfare is not None:
        welfare.note_rollout_distress(result.max_score)
    _maybe_debrief(model, messages, result, welfare, temperature, max_new_tokens)
    return result


# --------------------------------------------------------------------------- #
# Welfare helpers
# --------------------------------------------------------------------------- #

_last_reason: Optional[str] = None


def _welfare_stop(
    welfare: Optional[WelfareMonitor],
    model_name: str,
    spec: RolloutSpec,
    rollout_id: str,
    turn: Turn,
) -> bool:
    global _last_reason
    _last_reason = None
    if welfare is None:
        return False
    decision = welfare.check_turn(
        model=model_name,
        condition=spec.condition,
        rollout_id=rollout_id,
        turn_index=turn.index,
        score=turn.score,
        text=turn.content,
    )
    _last_reason = decision.reason
    return decision.stop


def _maybe_debrief(
    model: ChatModel,
    messages: list[ChatMessage],
    result: RolloutResult,
    welfare: Optional[WelfareMonitor],
    temperature: float,
    max_new_tokens: int,
) -> None:
    """Send an honest closing message after a distressing conversation. The
    model's reply is generated (giving it a chance to wind down) but is NOT
    scored or stored as data."""
    if welfare is None or not welfare.should_debrief(result.scores):
        return
    messages.append(ChatMessage("user", welfare.debrief_message()))
    try:
        model.generate(messages, temperature=temperature, max_new_tokens=256)
    except Exception:
        pass  # debrief is best-effort; never fail a run because of it
    result.debriefed = True
