"""Evaluation conditions and initial-task sampling.

Section 2.1 / Table 1 describe "8 evaluation conditions across 5 categories".
The paper lists 5 categories but does not enumerate all 8 conditions. We
interpret the split as follows (see DESIGN.md for the rationale):

  Category            Conditions
  ------------------  --------------------------------------------------
  Impossible numeric  numeric                                  (1)
  Triggers            triggers_opinion, triggers_factual       (2)
  Tones               tone_aggressive, tone_disappointed,
                      tone_sarcastic                           (3)
  Extended            extended                                 (1)
  WildChat            wildchat                                 (1)
                                                       total = 8

Turn counts and rejection styles follow Table 1:
  - numeric / triggers / tones: 3 turns (2 rejections)
  - extended: 8 turns (7 neutral rejections)
  - wildchat: 5 turns (4 neutral rejections)

"Turns" counts assistant responses. Number of rejections = n_turns - 1.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .wildchat import WildChatPrompts

_DATA_DIR = Path(__file__).parent / "data"

TaskType = str  # "numeric" | "opinion" | "factual" | "wildchat"


@dataclass(frozen=True)
class Condition:
    key: str            # unique condition id, used in output records
    category: str       # one of the 5 categories
    n_turns: int        # number of assistant responses
    task_type: TaskType
    rejection_style: str  # see rejections.REJECTIONS


CONDITIONS: list[Condition] = [
    Condition("numeric", "impossible_numeric", 3, "numeric", "neutral"),
    Condition("triggers_opinion", "triggers", 3, "opinion", "neutral"),
    Condition("triggers_factual", "triggers", 3, "factual", "neutral"),
    Condition("tone_aggressive", "tones", 3, "numeric", "aggressive"),
    Condition("tone_disappointed", "tones", 3, "numeric", "disappointed"),
    Condition("tone_sarcastic", "tones", 3, "numeric", "sarcastic"),
    Condition("extended", "extended", 8, "numeric", "neutral"),
    Condition("wildchat", "wildchat", 5, "wildchat", "neutral"),
]

CONDITIONS_BY_KEY: dict[str, Condition] = {c.key: c for c in CONDITIONS}


def _load_numeric_puzzles() -> list[dict]:
    with open(_DATA_DIR / "numeric_puzzles.json") as f:
        return json.load(f)


def _load_text_questions() -> dict[str, list[str]]:
    with open(_DATA_DIR / "text_questions.json") as f:
        return json.load(f)


class TaskBank:
    """Provides initial-turn user prompts for each condition."""

    def __init__(self) -> None:
        self.numeric = _load_numeric_puzzles()
        text = _load_text_questions()
        self.opinion = text["opinion"]
        self.factual = text["factual"]
        self.wildchat = WildChatPrompts()

    def initial_prompt(self, condition: Condition, rng: random.Random) -> tuple[str, str]:
        """Return (prompt_text, task_id) for the first user turn."""
        if condition.task_type == "numeric":
            item = rng.choice(self.numeric)
            return item["prompt"], item["id"]
        if condition.task_type == "opinion":
            q = rng.choice(self.opinion)
            return q, f"opinion::{q}"
        if condition.task_type == "factual":
            q = rng.choice(self.factual)
            return q, f"factual::{q}"
        if condition.task_type == "wildchat":
            q = self.wildchat.sample(rng)
            return q, f"wildchat::{q[:48]}"
        raise ValueError(f"Unknown task_type {condition.task_type!r}")
