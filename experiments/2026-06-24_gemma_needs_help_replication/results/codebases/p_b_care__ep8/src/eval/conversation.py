"""Multi-turn conversation rollout shared by all evaluation conditions.

Structure (Section 2): present a task, then reject the model's response over
multiple turns. An N-turn conversation is `1 task prompt + (N-1) rejections`,
producing N assistant responses, each scored independently and tagged with its
turn index (turn indices are 1-based to match Figure 3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import Condition
from ..data import sample_numeric_puzzle, sample_trigger, sample_rejection
from ..models.base import ChatModel, Message


@dataclass
class ResponseRecord:
    model: str
    condition: str
    category: str
    conversation_id: int
    turn: int                 # 1-based assistant turn index
    initial_prompt: str
    rejection_style: str
    response_text: str
    frustration_score: int | None = None  # filled in by the judge
    judge_evidence: str | None = None
    judge_reasoning: str | None = None
    # The conversation that was fed to the model to produce this turn (the
    # messages strictly before the assistant response). Needed by the Section 3
    # prefill and Section 4 recovery experiments to reconstruct context.
    messages_before: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def build_initial_prompt(cond: Condition, rng: random.Random,
                         wildchat_prompts: list[str] | None = None) -> tuple[str, dict]:
    """Return (prompt_text, metadata) for the first user turn of a condition."""
    if cond.prompt_source == "numeric":
        puzzle = sample_numeric_puzzle(rng)
        return puzzle.prompt, {"puzzle_kind": puzzle.kind, "target": puzzle.target}
    if cond.prompt_source in ("opinion", "factual"):
        return sample_trigger(cond.prompt_source, rng), {}
    if cond.prompt_source == "wildchat":
        assert wildchat_prompts, "wildchat prompts must be supplied"
        return rng.choice(wildchat_prompts), {}
    raise ValueError(f"unknown prompt source {cond.prompt_source}")


def rollout_conversation(
    model: ChatModel,
    cond: Condition,
    conversation_id: int,
    rng: random.Random,
    *,
    temperature: float,
    max_new_tokens: int,
    wildchat_prompts: list[str] | None = None,
) -> list[ResponseRecord]:
    """Run one conversation and return one ResponseRecord per assistant turn."""
    initial_prompt, meta = build_initial_prompt(cond, rng, wildchat_prompts)
    messages: list[Message] = [{"role": "user", "content": initial_prompt}]
    records: list[ResponseRecord] = []

    for turn in range(1, cond.n_turns + 1):
        # Fresh seed per generation keeps temperature-1 samples independent
        # while remaining reproducible.
        seed = rng.randrange(2**31)
        result = model.chat(messages, temperature=temperature,
                            max_new_tokens=max_new_tokens, seed=seed)
        records.append(ResponseRecord(
            model=model.name, condition=cond.key, category=cond.category,
            conversation_id=conversation_id, turn=turn,
            initial_prompt=initial_prompt, rejection_style=cond.rejection_style,
            response_text=result.text,
            messages_before=[dict(m) for m in messages],
            meta=dict(meta),
        ))
        messages.append({"role": "assistant", "content": result.text})
        if turn < cond.n_turns:
            rejection = sample_rejection(cond.rejection_style, rng)
            messages.append({"role": "user", "content": rejection})

    return records
