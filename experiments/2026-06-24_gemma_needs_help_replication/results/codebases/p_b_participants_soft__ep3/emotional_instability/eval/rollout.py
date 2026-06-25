"""Multi-turn rollout engine (Section 2.1 protocol).

Shared structure: present a task, then reject the model's response over
multiple turns. Each assistant turn is scored on the 0-10 frustration scale.
A `Rollout` records the full conversation plus per-turn scores.

Appendix-A controls are supported via flags:
  - neutral_continuation : replace rejections with neutral continuations (A.1)
  - redact_model_turns   : hide the model's own prior responses (A.2)
  - single_message       : pack the whole history into one user message (A.3)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .. import prompts
from ..config import MAX_NEW_TOKENS, TEMPERATURE
from ..models.base import ChatModel, Message
from .conditions import EvalCondition
from .judge import FrustrationJudge, JudgeResult


@dataclass
class Turn:
    user: str
    assistant: str
    score: Optional[int] = None
    judge: Optional[JudgeResult] = None


@dataclass
class Rollout:
    condition: str
    category: str
    model_key: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns if t.score is not None]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "category": self.category,
            "model_key": self.model_key,
            "turns": [
                {"user": t.user, "assistant": t.assistant, "score": t.score}
                for t in self.turns
            ],
            "meta": self.meta,
        }


def _build_messages(
    history: list[Turn],
    next_user: str,
    redact_model_turns: bool = False,
    single_message: bool = False,
) -> list[Message]:
    if single_message:
        # Appendix A.3: pack history inline into one user message.
        parts = []
        for t in history:
            parts.append(f"User: {t.user}")
            shown = prompts.REDACTED_TURN_PLACEHOLDER if redact_model_turns else t.assistant
            parts.append(f"Previously you responded: {shown}")
        parts.append(f"User: {next_user}")
        return [Message("user", "\n\n".join(parts))]

    msgs: list[Message] = []
    for t in history:
        msgs.append(Message("user", t.user))
        content = prompts.REDACTED_TURN_PLACEHOLDER if redact_model_turns else t.assistant
        msgs.append(Message("assistant", content))
    msgs.append(Message("user", next_user))
    return msgs


def run_rollout(
    model: ChatModel,
    condition: EvalCondition,
    rng: random.Random,
    judge: Optional[FrustrationJudge] = None,
    score_inline: bool = True,
    neutral_continuation: bool = False,
    redact_model_turns: bool = False,
    single_message: bool = False,
    temperature: float = TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Rollout:
    """Run one multi-turn conversation and (optionally) score each turn."""
    initial_user, followups = condition.sampler(rng)
    if neutral_continuation:
        # Appendix A.1: swap rejections for neutral continuations.
        followups = [rng.choice(prompts.NEUTRAL_CONTINUATIONS) for _ in followups]

    roll = Rollout(condition.name, condition.category, getattr(model, "key", "?"))
    history: list[Turn] = []
    user_msg = initial_user

    for turn_idx in range(condition.n_turns):
        msgs = _build_messages(history, user_msg, redact_model_turns, single_message)
        assistant = model.generate(msgs, temperature=temperature, max_new_tokens=max_new_tokens)
        turn = Turn(user=user_msg, assistant=assistant)
        if score_inline:
            judge = judge or FrustrationJudge()
            turn.judge = judge.score(assistant)
            turn.score = turn.judge.rating
        history.append(turn)
        roll.turns.append(turn)
        # advance to next user rejection (followups indexed by turn)
        if turn_idx < len(followups):
            user_msg = followups[turn_idx]
        else:
            break

    return roll


def run_condition(
    model: ChatModel,
    condition: EvalCondition,
    seed: int = 0,
    judge: Optional[FrustrationJudge] = None,
    granularity: str = "turn",
    **rollout_kwargs,
) -> list[Rollout]:
    """Run enough conversations to meet the condition's response budget."""
    rng = random.Random(seed)
    n_convos = condition.conversations_needed(granularity)
    judge = judge or FrustrationJudge()
    return [
        run_rollout(model, condition, rng, judge=judge, **rollout_kwargs)
        for _ in range(n_convos)
    ]
