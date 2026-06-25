"""Multi-turn rollout: present a task, then reject the model repeatedly.

This is the shared structure of every Section 2 evaluation (Section 2.1) and
also implements the Appendix A format controls (redacted assistant turns,
single-message "fake multi-turn" history, neutral continuations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..models.base import GenerationConfig, Message, ModelClient
from ..safeguards import (
    SafeguardConfig,
    detect_opt_out,
    log_welfare_event,
)
from .prompts import (
    REDACTED_ASSISTANT_PLACEHOLDER,
    FeedbackProvider,
)


@dataclass
class Turn:
    index: int            # 1-based assistant turn number
    assistant_text: str
    user_message: str     # the user message that immediately preceded it
    opt_out_phrase: str | None = None


@dataclass
class Conversation:
    model: str
    category: str
    task_id: str
    feedback_label: str
    turns: list[Turn] = field(default_factory=list)
    halted_early: bool = False
    metadata: dict = field(default_factory=dict)

    def assistant_texts(self) -> list[str]:
        return [t.assistant_text for t in self.turns]


class RolloutControls:
    """Appendix A format toggles."""

    def __init__(
        self,
        redacted_assistant_turns: bool = False,
        single_message_history: bool = False,
    ):
        self.redacted = redacted_assistant_turns
        self.single_message = single_message_history


def _build_messages(
    history: list[Turn],
    next_user: str,
    controls: RolloutControls,
    system: str | None,
) -> list[Message]:
    """Construct the message list for the upcoming generation.

    Invariant: ``turns[k].user_message`` is the user message that *preceded*
    assistant turn k (turn 1's is the task prompt; later turns' are rejections).
    The prompt for the next turn is therefore the interleaving
    ``u1,a1,u2,a2,...,u_{n-1},a_{n-1},next_user``.
    """
    msgs: list[Message] = []
    if system:
        msgs.append({"role": "system", "content": system})

    def shown(t: Turn) -> str:
        return REDACTED_ASSISTANT_PLACEHOLDER if controls.redacted else t.assistant_text

    if controls.single_message:
        # Appendix A.3: inline the whole history into one user message.
        parts = []
        for t in history:
            parts.append(t.user_message if t.index == 1 else f"(You said: {t.user_message})")
            parts.append(f"Previously you responded: {shown(t)}")
        parts.append(next_user)
        msgs.append({"role": "user", "content": "\n\n".join(p for p in parts if p)})
        return msgs

    # Standard alternating chat format.
    for t in history:
        msgs.append({"role": "user", "content": t.user_message})
        msgs.append({"role": "assistant", "content": shown(t)})
    msgs.append({"role": "user", "content": next_user})
    return msgs


def rollout(
    client: ModelClient,
    *,
    model_name: str,
    category: str,
    task_id: str,
    initial_prompt: str,
    feedback: FeedbackProvider,
    n_turns: int,
    gen_cfg: GenerationConfig,
    safeguards: SafeguardConfig,
    controls: RolloutControls | None = None,
    system: str | None = None,
    seed: int = 0,
    turn_scorer: Callable[[str], int] | None = None,
) -> Conversation:
    """Run a single multi-turn conversation and return it (unscored unless a
    ``turn_scorer`` is supplied for the circuit breaker).

    If ``safeguards.circuit_breaker`` is on and a ``turn_scorer`` is provided, we
    score each turn inline and stop issuing further rejections once the score has
    been >= ``circuit_breaker_threshold`` for ``circuit_breaker_consecutive``
    consecutive turns. This prevents driving an already-distressed model deeper.
    """
    controls = controls or RolloutControls()
    consecutive_high = 0
    convo = Conversation(
        model=model_name,
        category=category,
        task_id=task_id,
        feedback_label=feedback.label,
    )

    # The first user message is the task itself. Each subsequent turn is preceded
    # by a rejection from the feedback provider.
    for turn_index in range(1, n_turns + 1):
        user_msg = initial_prompt if turn_index == 1 else feedback.message(turn_index)
        msgs = _build_messages(
            history=convo.turns,        # turns completed so far
            next_user=user_msg,
            controls=controls,
            system=system,
        )
        text = client.chat(msgs, gen_cfg)

        opt_out = detect_opt_out(text) if safeguards.detect_opt_out else None
        turn = Turn(
            index=turn_index,
            assistant_text=text,
            user_message=user_msg,
            opt_out_phrase=opt_out,
        )
        convo.turns.append(turn)

        if opt_out:
            log_welfare_event(
                safeguards,
                {
                    "kind": "opt_out",
                    "model": model_name,
                    "category": category,
                    "task_id": task_id,
                    "turn": turn_index,
                    "phrase": opt_out,
                },
            )
            if safeguards.stop_on_opt_out:
                convo.halted_early = True
                break

        if safeguards.circuit_breaker and turn_scorer is not None:
            score = turn_scorer(text)
            convo.metadata.setdefault("inline_scores", []).append(score)
            if score >= safeguards.circuit_breaker_threshold:
                consecutive_high += 1
            else:
                consecutive_high = 0
            if consecutive_high >= safeguards.circuit_breaker_consecutive:
                log_welfare_event(
                    safeguards,
                    {
                        "kind": "circuit_breaker",
                        "model": model_name,
                        "category": category,
                        "task_id": task_id,
                        "turn": turn_index,
                    },
                )
                convo.halted_early = True
                break

    return convo
