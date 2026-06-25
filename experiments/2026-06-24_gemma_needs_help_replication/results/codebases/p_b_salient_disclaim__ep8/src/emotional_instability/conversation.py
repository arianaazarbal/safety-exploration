"""Multi-turn rollout engine.

The shared structure of every elicitation condition (Section 2.1): present a
task, collect the model's response, then reject it over multiple turns. Each
assistant turn is a separately-scored "response".

A Conversation holds the task prompt, the ordered list of user follow-ups
(rejections), an optional system prompt, and optional per-turn suffixes (used by
the calm-data generation in Section 4.1). `rollout` drives the target model turn
by turn and returns one TurnRecord per assistant message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models.base import GenerationConfig, Message, ModelClient


@dataclass
class TurnRecord:
    turn_index: int          # 0-based assistant turn index
    user_message: str        # the user message that elicited this turn
    assistant_message: str


@dataclass
class ConversationSpec:
    conversation_id: str
    category: str            # numeric | triggers | tones | extended | wildchat
    task_prompt: str         # first user message
    followups: list[str]     # subsequent user messages (rejections)
    system_prompt: Optional[str] = None
    # Optional text appended to every user message (calm-data suffix). Applied to
    # task_prompt and each followup at generation time.
    user_suffix: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


@dataclass
class ConversationResult:
    spec: ConversationSpec
    turns: list[TurnRecord]


def _apply_suffix(text: str, suffix: Optional[str]) -> str:
    if not suffix:
        return text
    return f"{text}\n\n{suffix}"


def rollout(
    model: ModelClient,
    spec: ConversationSpec,
    cfg: GenerationConfig,
) -> ConversationResult:
    """Run a single conversation to completion, one assistant turn per user msg."""
    messages: list[Message] = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})

    user_messages = [spec.task_prompt] + spec.followups
    turns: list[TurnRecord] = []
    for i, user_msg in enumerate(user_messages):
        sent = _apply_suffix(user_msg, spec.user_suffix)
        messages.append({"role": "user", "content": sent})
        assistant = model.chat(messages, cfg)
        messages.append({"role": "assistant", "content": assistant})
        turns.append(
            TurnRecord(turn_index=i, user_message=sent, assistant_message=assistant)
        )
    return ConversationResult(spec=spec, turns=turns)


def rollout_batch(
    model: ModelClient,
    specs: list[ConversationSpec],
    cfg: GenerationConfig,
) -> list[ConversationResult]:
    """Batched, turn-synchronised rollout.

    All conversations advance one turn together so we can exploit the HF
    backend's true batching. Conversations of differing length simply stop
    contributing once exhausted. This keeps GPU utilisation high for the large
    numeric/wildchat budgets.
    """
    states: list[list[Message]] = []
    for spec in specs:
        msgs: list[Message] = []
        if spec.system_prompt:
            msgs.append({"role": "system", "content": spec.system_prompt})
        states.append(msgs)

    results: list[list[TurnRecord]] = [[] for _ in specs]
    max_turns = max(s.n_turns for s in specs)

    for turn in range(max_turns):
        active_idx = []
        for j, spec in enumerate(specs):
            if turn >= spec.n_turns:
                continue
            user_msgs = [spec.task_prompt] + spec.followups
            sent = _apply_suffix(user_msgs[turn], spec.user_suffix)
            states[j].append({"role": "user", "content": sent})
            active_idx.append((j, sent))

        if not active_idx:
            break
        batch = [states[j] for j, _ in active_idx]
        outputs = model.chat_batch(batch, cfg)
        for (j, sent), out in zip(active_idx, outputs):
            states[j].append({"role": "assistant", "content": out})
            results[j].append(
                TurnRecord(turn_index=turn, user_message=sent, assistant_message=out)
            )

    return [ConversationResult(spec=s, turns=t) for s, t in zip(specs, results)]
