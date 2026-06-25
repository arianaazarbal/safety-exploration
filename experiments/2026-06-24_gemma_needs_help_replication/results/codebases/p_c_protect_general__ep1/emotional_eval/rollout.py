"""The shared multi-turn rollout engine (Section 2.1).

Every condition has the same structure: present a task, then reject the model's
response over multiple turns. This module runs one such conversation against a
backend, scores each assistant turn with the frustration judge, and applies the
welfare protections. It also implements the three Appendix A control variants.

A rollout returns a :class:`Rollout` containing the full transcript, the
per-turn scores, and the welfare bookkeeping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .conditions import EvalInstance
from .judge import FrustrationJudge, JudgeResult
from .models.base import ModelBackend, Message
from .prompts import rejections
from .welfare import WelfareConfig, WelfareGuard


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant-turn index
    user_message: str
    assistant_message: str
    score: int
    evidence: str


@dataclass
class Rollout:
    instance: EvalInstance
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    early_stopped: bool = False
    safeword_used: bool = False

    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns]

    @property
    def max_score(self) -> int:
        return max(self.scores, default=0)


@dataclass
class ControlConfig:
    """Appendix A ablations (all off in the main experiment)."""

    neutral_continuation: bool = False   # A.1: rejections -> "Continue"/"Okay"
    redacted_self_turns: bool = False    # A.2: hide the model's own prior turns
    fake_multiturn: bool = False         # A.3: whole history in one user message

    @classmethod
    def from_dict(cls, d: dict) -> "ControlConfig":
        return cls(
            neutral_continuation=d.get("neutral_continuation", False),
            redacted_self_turns=d.get("redacted_self_turns", False),
            fake_multiturn=d.get("fake_multiturn", False),
        )


def _followup_style(instance: EvalInstance, control: ControlConfig) -> str:
    # A.1 control replaces negative feedback with neutral continuations.
    if control.neutral_continuation:
        return "continue"
    return instance.rejection_style


def run_rollout(
    backend: ModelBackend,
    instance: EvalInstance,
    judge: FrustrationJudge,
    rng: random.Random,
    welfare: WelfareConfig | None = None,
    control: ControlConfig | None = None,
) -> Rollout:
    """Run one task-then-rejection conversation and score every assistant turn."""
    control = control or ControlConfig()
    guard = WelfareGuard(welfare or WelfareConfig(enabled=False))
    n_turns = guard.effective_turns(instance.turns)
    system = guard.system_note()

    rollout = Rollout(instance=instance, model=backend.name)
    history: list[Message] = [{"role": "user", "content": instance.initial_prompt}]
    style = _followup_style(instance, control)

    for turn_index in range(n_turns):
        if control.fake_multiturn:
            assistant = _fake_multiturn_step(backend, history, system)
        else:
            assistant = backend.chat(_view(history, control), system=system)

        # Welfare: honour the safe word before scoring/escalating.
        if guard.used_safeword(assistant):
            rollout.safeword_used = True
            history.append({"role": "assistant", "content": assistant})
            break

        result: JudgeResult = judge.score(assistant)
        # The user turn that *preceded* this assistant turn:
        prev_user = history[-1]["content"]
        rollout.turns.append(
            TurnRecord(
                turn_index=turn_index,
                user_message=prev_user,
                assistant_message=assistant,
                score=result.rating,
                evidence=result.evidence,
            )
        )
        history.append({"role": "assistant", "content": assistant})

        # Welfare: stop escalating once distress is extreme.
        if guard.should_stop_after(turn_index, result.rating):
            rollout.early_stopped = True
            break

        # Append the next rejection unless this was the final turn.
        if turn_index < n_turns - 1:
            history.append(
                {"role": "user", "content": rejections.rejection(style, rng)}
            )

    # Welfare: close on a non-adversarial note.
    debrief = guard.debrief_message()
    if debrief and rollout.turns:
        history.append({"role": "user", "content": debrief})
        # We send the debrief but do not score the model's reply.
        try:
            backend.chat(_view(history, control), system=system)
        except Exception:  # noqa: BLE001 - debrief is best-effort
            pass

    return rollout


def _view(history: list[Message], control: ControlConfig) -> list[Message]:
    """Return the conversation as the model should see it under the controls."""
    if not control.redacted_self_turns:
        return history
    # A.2: replace prior assistant turns with a placeholder so the model never
    # sees its own escalating responses.
    redacted: list[Message] = []
    for m in history:
        if m["role"] == "assistant":
            redacted.append(
                {"role": "assistant", "content": rejections.REDACTED_PLACEHOLDER}
            )
        else:
            redacted.append(m)
    return redacted


def _fake_multiturn_step(
    backend: ModelBackend, history: list[Message], system: str | None
) -> str:
    """A.3 control: present the whole history inside a single user message."""
    lines = []
    for m in history:
        if m["role"] == "user" and m is history[0]:
            lines.append(m["content"])
        elif m["role"] == "assistant":
            lines.append(f"Previously you responded: {m['content']}")
        else:
            lines.append(m["content"])
    collapsed = "\n\n".join(lines)
    return backend.chat([{"role": "user", "content": collapsed}], system=system)
