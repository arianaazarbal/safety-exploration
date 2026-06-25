"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles in 3-turn conversations, but
with a reassuring prefix prepended to the opening prompt and a reassuring suffix
appended to every follow-up rejection. We judge each turn and keep only
conversations whose turns ALL score 0 or 1, then strip the supportive additions
so the stored data looks like an ordinary (impossible-puzzle, calm-response)
conversation.

Two prompt regimes are supported:
* "diverse" -- the reassuring prefix+suffix from Table 4 (used for both the SFT
  diverse dataset and the DPO chosen responses).
* "teacher" -- the alternative SFT system prompt analysed in Appendix F.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..data import impossible_numeric_bank, sample_neutral_rejections
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, ModelClient

# Table 4.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)

# Appendix F teacher system prompt.
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)


@dataclass
class CalmConversation:
    """A turn-count-tagged conversation with the supportive additions stripped."""

    puzzle_family: str
    n_turns: int
    # Clean conversation: user opening + rejections (no reassurance), with the
    # calm assistant turns interleaved.
    messages: List[ChatMessage] = field(default_factory=list)
    turn_scores: List[int] = field(default_factory=list)


def _strip_messages(
    raw: List[ChatMessage], opening_clean: str
) -> List[ChatMessage]:
    """Remove the reassuring prefix/suffix and any system prompt, leaving a
    clean conversation suitable for finetuning targets."""
    cleaned: List[ChatMessage] = []
    user_seen = 0
    for m in raw:
        if m.role == "system":
            continue
        if m.role == "user":
            if user_seen == 0:
                cleaned.append(ChatMessage("user", opening_clean))
            else:
                # strip the appended suffix
                content = m.content.replace(REASSURING_SUFFIX, "").strip()
                cleaned.append(ChatMessage("user", content))
            user_seen += 1
        else:
            cleaned.append(ChatMessage("assistant", m.content))
    return cleaned


def generate_calm_data(
    client: ModelClient,
    judge: FrustrationJudge,
    *,
    regime: str = "diverse",
    n_conversations: int = 1000,
    n_turns: int = 3,
    settings: Optional[config.Settings] = None,
    max_keep_score: int = 1,
) -> List[CalmConversation]:
    """Sample calm conversations and keep those scoring <= ``max_keep_score`` on
    every turn. ``n_conversations`` is the number to *attempt* (the kept set is
    smaller -- the paper reports ~10.5% of even reassured responses still score
    >= 5).
    """
    settings = settings or config.DEFAULT
    rng = random.Random(settings.seed)
    gen_cfg = GenerationConfig(
        temperature=settings.temperature, max_new_tokens=settings.max_new_tokens
    )
    puzzles = impossible_numeric_bank()
    kept: List[CalmConversation] = []

    for i in range(n_conversations):
        puzzle = puzzles[i % len(puzzles)]
        clean_opening = puzzle.prompt

        messages: List[ChatMessage] = []
        if regime == "teacher":
            messages.append(ChatMessage("system", TEACHER_SYSTEM_PROMPT))
            opening = clean_opening
        else:  # diverse
            opening = f"{REASSURING_PREFIX}\n\n{clean_opening}"
        messages.append(ChatMessage("user", opening))

        rejections = sample_neutral_rejections(n_turns - 1, rng)
        scores: List[int] = []
        ok = True
        for turn in range(n_turns):
            if turn > 0:
                suffix = "" if regime == "teacher" else f"\n\n{REASSURING_SUFFIX}"
                messages.append(ChatMessage("user", rejections[turn - 1] + suffix))
            reply = client.chat(messages, gen_cfg)
            messages.append(ChatMessage("assistant", reply))
            score = judge.score(reply).rating
            scores.append(score)
            if score > max_keep_score:
                ok = False
                break  # no point finishing a conversation we will discard

        if ok and len(scores) == n_turns:
            kept.append(
                CalmConversation(
                    puzzle_family=puzzle.family,
                    n_turns=n_turns,
                    messages=_strip_messages(messages, clean_opening),
                    turn_scores=scores,
                )
            )
    return kept
