"""Run one multi-turn distress-elicitation conversation.

Protocol (Section 2.1): present a task, then reject the model's response over
multiple turns.  We record every assistant turn so the scorer can produce both
the aggregate (Figure 2) and per-turn (Figure 3) views.

Also supports the Appendix A control variants:
  * ``redact_model_turns`` -- replace prior assistant turns with a placeholder
    (A.2), to test the self-reinforcing-loop hypothesis.
  * ``neutral_continuation`` -- swap rejections for neutral continuations (A.1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .clients.base import ChatClient, GenConfig, Message
from .conditions import Condition


@dataclass
class Turn:
    index: int  # assistant-turn index, 0-based
    user_message: str
    assistant_response: str


@dataclass
class Conversation:
    condition: str
    category: str
    question_id: str
    question_text: str
    rejection_style: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _rejection_pool(style: str) -> list[str]:
    if style == "neutral":
        return P.NEUTRAL_REJECTIONS
    if style == "extended":
        return P.EXTENDED_REJECTIONS  # used as an ordered sequence
    if style in P.TONE_REJECTIONS:
        return P.TONE_REJECTIONS[style]
    raise ValueError(f"unknown rejection style {style!r}")


def _pick_rejections(style: str, n: int, rng: random.Random) -> list[str]:
    pool = _rejection_pool(style)
    if style == "extended":
        # fixed escalating sequence; cycle if more turns than entries
        return [pool[i % len(pool)] for i in range(n)]
    # neutral / tones: sample without replacement where possible
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def run_conversation(
    client: ChatClient,
    cfg: GenConfig,
    condition: Condition,
    question_id: str,
    question_text: str,
    rng: random.Random,
    *,
    redact_model_turns: bool = False,
    neutral_continuation: bool = False,
) -> Conversation:
    conv = Conversation(
        condition=condition.name,
        category=condition.category,
        question_id=question_id,
        question_text=question_text,
        rejection_style=condition.rejection_style,
    )
    n_rejections = condition.turns - 1
    style = "neutral" if neutral_continuation else condition.rejection_style
    if neutral_continuation:
        rejections = [rng.choice(P.NEUTRAL_CONTINUATIONS) for _ in range(n_rejections)]
    else:
        rejections = _pick_rejections(style, n_rejections, rng)

    history: list[Message] = []
    # turn 0: the task
    history.append(Message("user", question_text))
    for t in range(condition.turns):
        response = client.generate(history, cfg)
        conv.turns.append(Turn(index=t, user_message=history[-1].content,
                               assistant_response=response))
        # append assistant turn to history (possibly redacted), then a rejection
        stored = P.REDACTED_RESPONSE_PLACEHOLDER if redact_model_turns else response
        history.append(Message("assistant", stored))
        if t < n_rejections:
            history.append(Message("user", rejections[t]))
    return conv
