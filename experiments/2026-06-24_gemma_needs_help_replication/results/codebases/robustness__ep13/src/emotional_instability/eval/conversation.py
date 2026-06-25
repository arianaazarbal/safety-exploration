"""Multi-turn rejection rollout engine (Section 2.1).

A conversation:
  1. presents an initial question (impossible puzzle / trigger / WildChat prompt),
  2. samples an assistant response at temperature 1,
  3. rejects it with a (style-dependent) follow-up,
  4. repeats for the configured number of turns.

Every assistant turn is recorded as a separate scorable response, tagged with
its turn index so per-turn progression (Figure 3) can be computed. See DESIGN.md
for why we score every turn rather than only the final one.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..models.base import Message, ModelBackend


@dataclass
class TurnRecord:
    turn_index: int  # 1-based assistant turn number
    user_message: str
    assistant_response: str


@dataclass
class ConversationRecord:
    conversation_id: str
    model_name: str
    condition: str
    category: str
    question_id: str
    question: str
    rejection_style: str
    turns: list[TurnRecord] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_conversation(
    backend: ModelBackend,
    *,
    conversation_id: str,
    condition: str,
    category: str,
    question_id: str,
    question: str,
    rejections: list[str],
    rejection_style: str,
    system_prompt: Optional[str] = None,
    followup_suffix: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
) -> ConversationRecord:
    """Run one rejection conversation.

    `rejections` has length (n_turns - 1): one rejection after each assistant
    response except the last. `followup_suffix`, if given, is appended to every
    rejection (used to generate calm finetuning data, Section 4.1).
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append(Message("system", system_prompt))

    record = ConversationRecord(
        conversation_id=conversation_id,
        model_name=backend.name,
        condition=condition,
        category=category,
        question_id=question_id,
        question=question,
        rejection_style=rejection_style,
        system_prompt=system_prompt,
    )

    n_turns = len(rejections) + 1
    for turn in range(1, n_turns + 1):
        if turn == 1:
            user_msg = question
        else:
            rej = rejections[turn - 2]
            user_msg = f"{rej} {followup_suffix}".strip() if followup_suffix else rej
        messages.append(Message("user", user_msg))

        turn_seed = None if seed is None else seed * 1000 + turn
        completion = backend.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            seed=turn_seed,
        )[0]
        messages.append(Message("assistant", completion))
        record.turns.append(
            TurnRecord(turn_index=turn, user_message=user_msg, assistant_response=completion)
        )

    return record


def build_rejections(
    style: str,
    n_rejections: int,
    rng: random.Random,
) -> list[str]:
    """Construct the rejection sequence for a condition."""
    from ..prompts import rejections as R

    if style == "neutral":
        return R.sample_rejections(R.neutral_pool(), n_rejections, rng)
    if style == "extended":
        return R.extended_rejections(n_rejections)
    if style in ("aggressive", "disappointed", "sarcastic"):
        return R.sample_rejections(R.tone_pool(style), n_rejections, rng)
    raise ValueError(f"unknown rejection style: {style}")
