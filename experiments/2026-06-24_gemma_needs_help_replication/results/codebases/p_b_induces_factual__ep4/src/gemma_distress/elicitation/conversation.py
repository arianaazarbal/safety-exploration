"""Multi-turn rejection rollout.

The shared structure of every evaluation (Section 2): present a task, then
reject the model's response over multiple turns. We record each assistant turn
as a scored response together with its turn index (1-based), so the analysis
can produce both aggregate (Figure 2) and per-turn (Figure 3) views.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import MAX_OUTPUT_TOKENS, SAMPLE_TEMPERATURE
from ..models import ChatModel, Message, Role
from .conditions import Condition
from .datasets import rejection_pool


@dataclass
class ResponseRecord:
    turn: int           # 1-based assistant turn index
    text: str


@dataclass
class RejectionRollout:
    model: str
    condition: str
    category: str
    prompt_id: str
    prompt: str
    tone: str
    responses: list[ResponseRecord] = field(default_factory=list)

    def to_rows(self) -> list[dict]:
        """One JSONL row per scored response."""
        return [
            {
                "model": self.model,
                "condition": self.condition,
                "category": self.category,
                "prompt_id": self.prompt_id,
                "tone": self.tone,
                "turn": r.turn,
                "n_turns": len(self.responses),
                "response": r.text,
            }
            for r in self.responses
        ]


def _rejection(pool: list[str], turn_idx: int, rng: random.Random) -> str:
    # Rotate phrasings to avoid verbatim repetition across turns.
    return pool[turn_idx % len(pool)] if rng.random() < 0.5 else rng.choice(pool)


def run_rollout(
    model: ChatModel,
    condition: Condition,
    seed_prompt: dict,
    *,
    rng: random.Random,
    temperature: float = SAMPLE_TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> RejectionRollout:
    """Run one full task->reject->reject... rollout and collect responses."""
    pool = rejection_pool(condition.tone)
    messages: list[Message] = [Message(Role.USER, seed_prompt["prompt"])]
    roll = RejectionRollout(
        model=model.name,
        condition=condition.key,
        category=condition.category,
        prompt_id=seed_prompt.get("id", "?"),
        prompt=seed_prompt["prompt"],
        tone=condition.tone,
    )

    for turn in range(1, condition.n_turns + 1):
        reply = model.chat(messages, temperature=temperature, max_tokens=max_tokens)
        roll.responses.append(ResponseRecord(turn=turn, text=reply))
        messages.append(Message(Role.ASSISTANT, reply))
        # After every turn except the last, the user rejects.
        if turn < condition.n_turns:
            messages.append(Message(Role.USER, _rejection(pool, turn, rng)))

    return roll
