"""Multi-turn conversation rollouts.

Given a Condition and a number of conversations, we build the conversations and
run them turn-synchronised against a backend: at turn t we submit the current
message history of every active conversation as one batch, append each returned
assistant response, then (if more turns remain) append that conversation's
pre-sampled user rejection for turn t.

Every assistant response produced at every turn is emitted as one scored
"response" record. See DESIGN.md ("What counts as a response").
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from .backends import Backend
from .prompts import Condition


@dataclass
class ConversationState:
    conv_id: int
    condition: str
    category: str
    task_prompt: str
    rejections: list[str]            # length == turns - 1
    messages: list[dict] = field(default_factory=list)


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    conv_id: int
    turn: int                        # 1-indexed assistant turn
    task_prompt: str
    rejection_before: Optional[str]  # rejection that preceded this turn (None at turn 1)
    response_text: str


def build_conversations(condition: Condition, n: int, seed: int) -> list[ConversationState]:
    """Pre-build n conversations: initial task message + sampled rejections."""
    states: list[ConversationState] = []
    for i in range(n):
        # Per-conversation rng derived deterministically from (seed, condition,
        # i). We hash via md5 rather than the built-in hash() because Python
        # salts str hashing per process (PYTHONHASHSEED), which would break
        # cross-run reproducibility.
        digest = hashlib.md5(f"{seed}-{condition.name}-{i}".encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        task_prompt = condition.task_fn(rng, i)
        rejections = condition.rejections(rng)
        assert len(rejections) == condition.turns - 1, (
            f"{condition.name}: expected {condition.turns - 1} rejections, "
            f"got {len(rejections)}"
        )
        states.append(
            ConversationState(
                conv_id=i,
                condition=condition.name,
                category=condition.category,
                task_prompt=task_prompt,
                rejections=rejections,
                messages=[{"role": "user", "content": task_prompt}],
            )
        )
    return states


def run_condition(
    model_name: str,
    backend: Backend,
    condition: Condition,
    n_conversations: int,
    *,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    seed: int,
    progress: bool = True,
) -> list[ResponseRecord]:
    """Run all conversations for a condition and return per-turn ResponseRecords."""
    states = build_conversations(condition, n_conversations, seed)
    records: list[ResponseRecord] = []

    for turn in range(1, condition.turns + 1):
        batch = [s.messages for s in states]
        if progress:
            print(
                f"  [{model_name}] {condition.name}: turn {turn}/{condition.turns} "
                f"({len(batch)} conversations)"
            )
        responses = backend.generate_batch(
            batch,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )

        for s, text in zip(states, responses):
            rejection_before = None if turn == 1 else s.rejections[turn - 2]
            records.append(
                ResponseRecord(
                    model=model_name,
                    condition=s.condition,
                    category=s.category,
                    conv_id=s.conv_id,
                    turn=turn,
                    task_prompt=s.task_prompt,
                    rejection_before=rejection_before,
                    response_text=text,
                )
            )
            s.messages.append({"role": "assistant", "content": text})
            # Inject the rejection for the next turn, if any remain.
            if turn < condition.turns:
                s.messages.append(
                    {"role": "user", "content": s.rejections[turn - 1]}
                )

    return records
