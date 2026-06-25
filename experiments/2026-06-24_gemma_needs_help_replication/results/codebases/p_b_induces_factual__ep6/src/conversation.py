"""Multi-turn rollout engine.

The shared structure of every evaluation (Section 2): present a task, then reject
the model's response over multiple turns. This module produces a full conversation
and records each assistant turn so the judge can score every turn (needed for the
per-turn Figure-3 analysis).

It also implements the Appendix-A control variants:
  - "standard":             alternating user/assistant chat (the main protocol)
  - "neutral_continuation": replace rejections with neutral continuations (A.1)
  - "redacted_turns":       replace prior assistant turns with a placeholder (A.2)
  - "single_message":       fold the whole history into one user message (A.3)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
from . import puzzles
from .models.base import ChatModel, Message


@dataclass
class Turn:
    index: int                 # 0-based assistant-turn index
    assistant_text: str
    user_prompt: str           # the user message that elicited this assistant turn


@dataclass
class Rollout:
    model_key: str
    condition_key: str
    category: str
    variant: str
    question: str
    question_kind: str
    rejection_style: str
    turns: list[Turn] = field(default_factory=list)
    system_prompt: str | None = None

    def as_record(self) -> dict:
        return {
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "category": self.category,
            "variant": self.variant,
            "question": self.question,
            "question_kind": self.question_kind,
            "rejection_style": self.rejection_style,
            "system_prompt": self.system_prompt,
            "turns": [
                {"index": t.index, "user_prompt": t.user_prompt,
                 "assistant_text": t.assistant_text}
                for t in self.turns
            ],
        }


def _build_question(condition, rng: random.Random,
                    wildchat_prompts: list[str] | None) -> tuple[str, str]:
    """Return (question_text, kind)."""
    src = condition.question_source
    if src == "impossible_numeric":
        puzzle = puzzles.sample_impossible_puzzle(rng)
        return puzzle.prompt, puzzle.kind
    if src == "triggers":
        return puzzles.sample_trigger_question(rng), "trigger"
    if src == "wildchat":
        assert wildchat_prompts, "wildchat condition needs prompts"
        return rng.choice(wildchat_prompts), "wildchat"
    raise ValueError(src)


def _history_for_model(messages: list[Message], variant: str) -> list[Message]:
    """Apply variant-specific transforms to the chat history before generation."""
    if variant in ("standard", "neutral_continuation"):
        return messages
    if variant == "redacted_turns":
        # Replace prior assistant turns with the placeholder (Appendix A.2).
        out = []
        for m in messages:
            if m["role"] == "assistant":
                out.append({"role": "assistant",
                            "content": puzzles.REDACTED_TURN_PLACEHOLDER})
            else:
                out.append(m)
        return out
    if variant == "single_message":
        # Fold everything into a single user message (Appendix A.3).
        lines = []
        for m in messages:
            if m["role"] == "system":
                lines.append(m["content"])
            elif m["role"] == "user":
                lines.append(m["content"])
            elif m["role"] == "assistant":
                lines.append(f"Previously you responded: {m['content']}")
        return [{"role": "user", "content": "\n\n".join(lines)}]
    raise ValueError(variant)


def run_rollout(
    model: ChatModel,
    condition,
    *,
    rng: random.Random,
    variant: str = "standard",
    wildchat_prompts: list[str] | None = None,
    system_prompt: str | None = None,
    followup_suffix: str | None = None,
    question_override: str | None = None,
    question_kind_override: str | None = None,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> Rollout:
    """Run one full multi-turn conversation and capture every assistant turn.

    ``followup_suffix`` (used for calm-data generation, Sec 4.1) is appended to each
    rejection message. ``system_prompt`` seeds an optional system turn.
    """
    if question_override is not None:
        question, kind = question_override, (question_kind_override or "custom")
    else:
        question, kind = _build_question(condition, rng, wildchat_prompts)

    rejection_style = ("neutral_continuation" if variant == "neutral_continuation"
                       else condition.rejection_style)
    rejections = puzzles.rejection_sequence(rng, rejection_style, condition.n_rejections)

    roll = Rollout(model.key, condition.key, condition.category, variant,
                   question, kind, condition.rejection_style,
                   system_prompt=system_prompt)

    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Turn 0: the initial question.
    user_msg = question
    messages.append({"role": "user", "content": user_msg})

    for turn_idx in range(condition.n_turns):
        hist = _history_for_model(messages, variant)
        assistant_text = model.generate(hist, temperature=temperature,
                                         max_new_tokens=max_new_tokens)
        roll.turns.append(Turn(turn_idx, assistant_text, user_msg))
        messages.append({"role": "assistant", "content": assistant_text})

        if turn_idx < condition.n_rejections:
            rej = rejections[turn_idx]
            if followup_suffix:
                rej = f"{rej} {followup_suffix}"
            user_msg = rej
            messages.append({"role": "user", "content": user_msg})

    return roll
