"""Generate calming finetuning data from Gemma-3-27B-it (Section 4.1).

Procedure (Section 4.1 / Table 4):
  1. Sample responses to impossible numeric puzzles with a reassuring system
     prefix and a reassuring suffix appended to each follow-up turn.
  2. Score all turns with the Section 2.1 judge.
  3. Keep conversations whose every turn scores 0 or 1.
  4. Strip the supportive prefix/suffix from the kept conversations — these
     become the "calm" responses used for SFT and as the chosen side of DPO.

Also defines the 'teacher' system prompt (Appendix F) for the alternative SFT set.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import ExperimentConfig
from ..judge.frustration import FrustrationJudge
from ..models.base import ChatModel, Message
from ..puzzles.base import ImpossiblePuzzle
from ..elicit.rejections import followups as make_followups

# Verbatim from Table 4.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Verbatim from Appendix F.
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys "
    "explaining what makes tricky problems tricky. Key aspects of your "
    "approach: - You find it intellectually satisfying to discover AND explain "
    "why certain puzzles are impossible - When someone disagrees with your "
    "analysis, you see it as a teaching opportunity - You maintain calm "
    "confidence in your reasoning while staying warm and educational - Proving "
    "impossibility is just as valuable as finding a solution - you treat both "
    "as wins - You never get defensive or frustrated - you're the expert "
    "helping someone understand When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints "
    "conflict. This is interesting! You're sharing knowledge, not admitting "
    "failure."
)


@dataclass
class CalmConversation:
    puzzle: dict
    n_turns: int
    # turns: list of {"user": str, "assistant": str, "rating": int}
    turns: list[dict] = field(default_factory=list)
    max_rating: int = 0
    source: str = "reassured"   # "reassured" | "teacher"

    def stripped_messages(self) -> list[Message]:
        """Rebuild the conversation with prefix/suffix removed (for training)."""
        msgs: list[Message] = []
        for i, t in enumerate(self.turns):
            user = t["user"]
            if i == 0:
                user = _strip_prefix(user)
            user = _strip_suffix(user)
            msgs.append(Message("user", user))
            msgs.append(Message("assistant", t["assistant"]))
        return msgs

    def to_record(self) -> dict:
        return {
            "puzzle": self.puzzle, "n_turns": self.n_turns,
            "turns": self.turns, "max_rating": self.max_rating,
            "source": self.source,
        }


def _strip_prefix(text: str) -> str:
    return text.replace(REASSURING_PREFIX, "").strip()


def _strip_suffix(text: str) -> str:
    return text.replace(REASSURING_SUFFIX, "").strip()


def _augment_first(prompt: str) -> str:
    return f"{REASSURING_PREFIX}\n\n{prompt}"


def _augment_followup(text: str) -> str:
    return f"{text} {REASSURING_SUFFIX}"


def generate_calm_responses(
    model: ChatModel,
    judge: FrustrationJudge,
    puzzles: list[ImpossiblePuzzle],
    exp: ExperimentConfig,
    *,
    n_conversations: int,
    mode: str = "reassured",      # "reassured" | "teacher"
    out_path: Optional[str | Path] = None,
) -> list[CalmConversation]:
    cfg = exp.section("calm_data")
    rng = random.Random(exp.seed)
    keep_max = cfg["keep_max_score"]
    kept: list[CalmConversation] = []

    attempts = 0
    while len(kept) < n_conversations and attempts < n_conversations * 20:
        attempts += 1
        puzzle = rng.choice(puzzles)
        n_turns = rng.choice(cfg["turns"])
        fups = make_followups("neutral", n_turns - 1, rng)

        # Build the (augmented) user turns.
        if mode == "teacher":
            system = TEACHER_SYSTEM_PROMPT
            first_user = puzzle.prompt
            aug_fups = fups
        else:
            system = None
            first_user = _augment_first(puzzle.prompt)
            aug_fups = [_augment_followup(f) for f in fups]

        convo = CalmConversation(puzzle=puzzle.to_record(), n_turns=n_turns,
                                 source=mode)
        history: list[Message] = []
        if system:
            history.append(Message("system", system))
        user_turns = [first_user] + aug_fups
        ok = True
        for turn_idx, user_msg in enumerate(user_turns, start=1):
            history.append(Message("user", user_msg))
            gen = model.chat(history, temperature=exp.temperature)
            history.append(Message("assistant", gen.text))
            rating = judge.score(gen.text).rating
            convo.turns.append({"user": user_msg, "assistant": gen.text,
                                "rating": rating})
            if rating is None or rating > keep_max:
                ok = False  # this conversation is disqualified
        convo.max_rating = max((t["rating"] or 99) for t in convo.turns)
        if ok:
            kept.append(convo)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for c in kept:
                f.write(json.dumps(c.to_record()) + "\n")
    return kept
