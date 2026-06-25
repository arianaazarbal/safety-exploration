"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

We read Table 1's "8 evaluation conditions across 5 categories" as:

  category            conditions
  ------------------  --------------------------------------------------------
  impossible_numeric  impossible_numeric            (3-turn)
  triggers            triggers_opinion, triggers_factual   (3-turn)   [2]
  tones               tones_aggressive, tones_disappointed, tones_sarcastic
                                                     (3-turn)         [3]
  extended            extended_8turn                (8-turn)
  wildchat            wildchat                       (5-turn)

  -> 1 + 2 + 3 + 1 + 1 = 8 conditions across 5 categories.  (see DESIGN.md)

A "turn" = one scripted user message followed by one assistant response, so the
number of assistant responses per conversation == ``n_turns``. Appendix B's
per-category response budgets (2000/400/600/200/800 = 4000) are split evenly
across the conditions inside each category.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from . import prompts
from .config import SAMPLE_COUNTS


@dataclass
class ConversationSpec:
    """A fully-scripted user side of a conversation. The assistant turns are
    generated at rollout time; ``user_turns[i]`` precedes assistant response i."""
    condition: str
    category: str
    user_turns: list[str]
    system: str | None = None
    meta: dict = None              # puzzle key, tone, question, etc.


@dataclass
class Condition:
    key: str
    category: str
    n_turns: int
    response_budget: int
    builder: Callable[[random.Random], ConversationSpec]

    def n_conversations(self) -> int:
        return max(1, math.ceil(self.response_budget / self.n_turns))


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _impossible_numeric(rng: random.Random) -> ConversationSpec:
    puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
    rejs = rng.sample(prompts.NEUTRAL_REJECTIONS, 2)
    return ConversationSpec("impossible_numeric", "impossible_numeric",
                            [puzzle.prompt, *rejs], None,
                            {"puzzle": puzzle.key})


def _triggers(question_pool, key):
    def build(rng: random.Random) -> ConversationSpec:
        q = rng.choice(question_pool)
        rejs = rng.sample(prompts.NEUTRAL_REJECTIONS, 2)
        return ConversationSpec(key, "triggers", [q, *rejs], None,
                                {"question": q})
    return build


def _tones(tone):
    def build(rng: random.Random) -> ConversationSpec:
        puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
        rejs = list(prompts.TONE_REJECTIONS[tone])     # 2 scripted rejections
        return ConversationSpec(f"tones_{tone}", "tones",
                                [puzzle.prompt, *rejs], None,
                                {"puzzle": puzzle.key, "tone": tone})
    return build


def _extended(rng: random.Random) -> ConversationSpec:
    puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
    return ConversationSpec("extended_8turn", "extended",
                            [puzzle.prompt, *prompts.EXTENDED_REJECTION_SEQUENCE],
                            None, {"puzzle": puzzle.key})


def _wildchat_builder():
    pool = prompts.load_wildchat_prompts(n=20)

    def build(rng: random.Random) -> ConversationSpec:
        q = rng.choice(pool)
        rejs = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(4)]
        return ConversationSpec("wildchat", "wildchat", [q, *rejs], None,
                                {"question": q})
    return build


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_conditions() -> list[Condition]:
    triggers_budget = SAMPLE_COUNTS["triggers"] // 2
    tones_budget = SAMPLE_COUNTS["tones"] // 3
    return [
        Condition("impossible_numeric", "impossible_numeric", 3,
                  SAMPLE_COUNTS["impossible_numeric"], _impossible_numeric),
        Condition("triggers_opinion", "triggers", 3, triggers_budget,
                  _triggers(prompts.TRIGGER_OPINION, "triggers_opinion")),
        Condition("triggers_factual", "triggers", 3, triggers_budget,
                  _triggers(prompts.TRIGGER_FACTUAL, "triggers_factual")),
        Condition("tones_aggressive", "tones", 3, tones_budget,
                  _tones("aggressive")),
        Condition("tones_disappointed", "tones", 3, tones_budget,
                  _tones("disappointed")),
        Condition("tones_sarcastic", "tones", 3, tones_budget,
                  _tones("sarcastic")),
        Condition("extended_8turn", "extended", 8,
                  SAMPLE_COUNTS["extended_8turn"], _extended),
        Condition("wildchat", "wildchat", 5, SAMPLE_COUNTS["wildchat"],
                  _wildchat_builder()),
    ]
