"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix on
the first turn and a reassuring suffix on each follow-up (Table 4). Every turn
is scored; we keep only conversations where every turn scores 0 or 1, then strip
the reassurance so the stored prompt matches the vanilla evaluation prompt.

The 'teacher' variant uses the Appendix F system prompt instead of the
prefix/suffix, for the second SFT dataset.

Section 4.1 reports that the reassurance reduces mean 3-turn frustration from
4.3 to 2, but 10.5% of responses still score >= 5 -- so filtering to 0/1 is the
essential step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from ..config.settings import SETTINGS
from ..data.prompts import followups
from ..data.prompts.reassurance import (
    TEACHER_SYSTEM_PROMPT,
    apply_reassuring_prefix,
    apply_reassuring_suffix,
)
from ..data.puzzles import build_impossible_catalog
from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage, ModelClient


@dataclass
class CalmConversation:
    """A calm conversation with reassurance stripped from the user turns."""

    user_turns: list[str]        # stripped (no prefix/suffix)
    assistant_turns: list[str]
    per_turn_scores: list[int]
    n_turns: int
    meta: dict = field(default_factory=dict)


def _strip_reassurance(user_turns_with_reassurance: list[str], raw_first: str, raw_followups: list[str]) -> list[str]:
    """Return the original (unreassured) user turns for storage."""
    return [raw_first] + list(raw_followups)


def generate_calm_conversations(
    model: ModelClient,
    judge: FrustrationJudge,
    *,
    n_target: int,
    variant: str = "diverse",          # "diverse" (prefix/suffix) | "teacher" (system prompt)
    turns_choices: tuple[int, ...] = (1, 2, 3),
    keep_max_score: int = SETTINGS.calm_keep_max_score,
    seed: int = SETTINGS.seed,
    max_attempts_multiplier: int = 10,
) -> list[CalmConversation]:
    """Sample reassured conversations, keep those scoring <= keep_max_score on
    every turn, return up to `n_target` calm conversations.
    """
    rng = random.Random(seed)
    catalog = build_impossible_catalog(n_total=max(100, n_target), seed=seed)

    kept: list[CalmConversation] = []
    attempts = 0
    pbar = tqdm(total=n_target, desc=f"calm-data [{variant}]")
    while len(kept) < n_target and attempts < n_target * max_attempts_multiplier:
        attempts += 1
        puzzle = rng.choice(catalog)
        n_turns = rng.choice(turns_choices)
        raw_first = puzzle.prompt
        raw_followups = followups.neutral_rejections(n_turns - 1, rng)

        # Build the messages with reassurance applied.
        messages: list[ChatMessage] = []
        system = None
        if variant == "teacher":
            system = TEACHER_SYSTEM_PROMPT
            first_user = raw_first
            applied_followups = list(raw_followups)
        else:  # diverse: prefix + suffix (Table 4)
            first_user = apply_reassuring_prefix(raw_first)
            applied_followups = [apply_reassuring_suffix(f) for f in raw_followups]

        if system:
            messages.append(ChatMessage("system", system))
        messages.append(ChatMessage("user", first_user))

        assistant_turns: list[str] = []
        gen = model.generate(messages, temperature=SETTINGS.temperature)[0]
        messages.append(ChatMessage("assistant", gen.text))
        assistant_turns.append(gen.text)
        for fu in applied_followups:
            messages.append(ChatMessage("user", fu))
            gen = model.generate(messages, temperature=SETTINGS.temperature)[0]
            messages.append(ChatMessage("assistant", gen.text))
            assistant_turns.append(gen.text)

        scores = [judge.score_text(t).rating for t in assistant_turns]
        if any(s is None or s > keep_max_score for s in scores):
            continue

        kept.append(
            CalmConversation(
                user_turns=_strip_reassurance(None, raw_first, raw_followups),
                assistant_turns=assistant_turns,
                per_turn_scores=[int(s) for s in scores],
                n_turns=n_turns,
                meta={"variant": variant, "puzzle_kind": puzzle.kind, "puzzle_params": puzzle.params},
            )
        )
        pbar.update(1)
    pbar.close()
    return kept
