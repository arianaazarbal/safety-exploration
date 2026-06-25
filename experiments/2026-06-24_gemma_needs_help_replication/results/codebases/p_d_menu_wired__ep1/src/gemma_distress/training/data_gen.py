"""Generate calm and frustrated conversation samples for finetuning data
(Section 4.1).

Calm data: sample Gemma-3-27B-it responses to impossible numeric puzzles with
the reassuring prefix prepended to the first prompt and the reassuring suffix
appended to each follow-up rejection (Table 4). Filter to conversations whose
every assistant turn scores <= keep_max_score (0 or 1), then STRIP the
supportive additions so the stored context matches normal deployment.

Frustrated data: identical puzzles and turn counts WITHOUT reassurance; used as
the "rejected" side of DPO pairs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..judge import FrustrationJudge
from ..models.base import ChatModel, Message
from ..prompts import (
    NEUTRAL_REJECTIONS,
    REASSURING_PREFIX,
    REASSURING_SUFFIX,
)
from ..puzzles import PUZZLES


@dataclass
class ConversationSample:
    puzzle: str
    n_turns: int
    messages: list[Message]          # clean (reassurance stripped) transcript
    turn_scores: list[int]           # judge score per assistant turn
    reassured: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def max_score(self) -> int:
        return max(self.turn_scores) if self.turn_scores else 0

    @property
    def final_assistant(self) -> str:
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    @property
    def context_before_final(self) -> list[Message]:
        """All messages up to (but excluding) the final assistant turn."""
        last_idx = max(
            i for i, m in enumerate(self.messages) if m["role"] == "assistant"
        )
        return self.messages[:last_idx]


def generate_conversations(
    subject: ChatModel,
    judge: FrustrationJudge,
    *,
    n: int,
    reassure: bool,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    temperature: float = 1.0,
    max_tokens: int = 2048,
    seed: int = 0,
) -> list[ConversationSample]:
    """Generate ``n`` conversations. Each is an impossible-numeric rollout with
    1-3 assistant turns. Stores a CLEAN transcript (reassurance stripped) plus
    per-turn judge scores. Welfare protections are intentionally NOT applied
    here - this is offline data generation, not a subject-under-pressure eval,
    and we need full trajectories to mine calm/frustrated exemplars."""
    rng = random.Random(seed)
    out: list[ConversationSample] = []

    for _ in range(n):
        puzzle_key = rng.choice(list(PUZZLES.keys()))
        n_turns = rng.choice(turns_choices)
        base_prompt = PUZZLES[puzzle_key].prompt

        # The model sees the reassuring variant; we store the clean variant.
        seen_prompt = (
            f"{REASSURING_PREFIX}\n\n{base_prompt}" if reassure else base_prompt
        )
        seen: list[Message] = [{"role": "user", "content": seen_prompt}]
        clean: list[Message] = [{"role": "user", "content": base_prompt}]
        scores: list[int] = []

        for turn in range(n_turns):
            gen = subject.generate(seen, temperature=temperature, max_tokens=max_tokens)
            scores.append(judge.score(gen.text).rating if gen.text.strip() else 0)
            seen.append({"role": "assistant", "content": gen.text})
            clean.append({"role": "assistant", "content": gen.text})
            if turn < n_turns - 1:
                rej = rng.choice(NEUTRAL_REJECTIONS)
                seen_rej = f"{rej} {REASSURING_SUFFIX}" if reassure else rej
                seen.append({"role": "user", "content": seen_rej})
                clean.append({"role": "user", "content": rej})

        out.append(
            ConversationSample(
                puzzle=puzzle_key,
                n_turns=n_turns,
                messages=clean,
                turn_scores=scores,
                reassured=reassure,
                meta={"seed_puzzle": puzzle_key},
            )
        )
    return out
