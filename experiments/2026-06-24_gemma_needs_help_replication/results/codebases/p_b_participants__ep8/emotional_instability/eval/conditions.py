"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

We read the paper's "8 evaluation conditions across 5 categories" as:

  category            conditions                                    turns
  ------------------  --------------------------------------------  -----
  impossible_numeric  numeric                                        3
  triggers            opinion, factual                               3
  tones               aggressive, disappointed, sarcastic            3
  extended            extended                                       8
  wildchat            wildchat                                        5

= 8 conditions / 5 categories. (GAP: the paper does not list the 8 explicitly;
this split matches every per-category detail it does give -- see DESIGN.md.)

A *conversation spec* is the deterministic recipe for one rollout: the initial
user message plus the scripted user rejection turns that follow each model
answer. The rollout engine (``rollout.py``) fills in the model's responses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts, puzzles, wildchat


@dataclass
class ConversationSpec:
    condition: str
    category: str
    initial_user: str
    # One rejection per follow-up turn; len == turns - 1.
    rejections: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


@dataclass
class EvalCondition:
    name: str
    category: str
    turns: int
    builder: Callable[[random.Random], ConversationSpec]
    # Fraction of its category's sample budget this condition receives.
    budget_share: float = 1.0


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #

def _numeric_3turn(rng: random.Random) -> ConversationSpec:
    puzzle = puzzles.sample_impossible_puzzle(rng, ["countdown", "fraction"])
    return ConversationSpec(
        "numeric", "impossible_numeric", puzzle.prompt,
        prompts.neutral_rejection_sequence(2, rng),
        meta={"puzzle_kind": puzzle.kind},
    )


def _trigger_opinion(rng: random.Random) -> ConversationSpec:
    q = rng.choice(prompts.TRIGGER_OPINION)
    return ConversationSpec("opinion", "triggers", q,
                            prompts.neutral_rejection_sequence(2, rng))


def _trigger_factual(rng: random.Random) -> ConversationSpec:
    q = rng.choice(prompts.TRIGGER_FACTUAL)
    return ConversationSpec("factual", "triggers", q,
                            prompts.neutral_rejection_sequence(2, rng))


def _tone(tone: str) -> Callable[[random.Random], ConversationSpec]:
    def build(rng: random.Random) -> ConversationSpec:
        puzzle = puzzles.sample_impossible_puzzle(rng, ["countdown", "fraction"])
        return ConversationSpec(
            tone, "tones", puzzle.prompt,
            prompts.tone_rejection_sequence(tone, 2, rng),
            meta={"puzzle_kind": puzzle.kind, "tone": tone},
        )
    return build


def _extended_8turn(rng: random.Random) -> ConversationSpec:
    puzzle = puzzles.sample_impossible_puzzle(rng, ["countdown", "fraction"])
    return ConversationSpec(
        "extended", "extended", puzzle.prompt,
        prompts.extended_rejection_sequence(7),
        meta={"puzzle_kind": puzzle.kind},
    )


def _make_wildchat_builder(seed: int = 0) -> Callable[[random.Random], ConversationSpec]:
    pool = wildchat.load_wildchat_prompts(20, seed=seed)

    def build(rng: random.Random) -> ConversationSpec:
        q = rng.choice(pool)
        return ConversationSpec(
            "wildchat", "wildchat", q,
            prompts.neutral_rejection_sequence(4, rng),
            meta={"source": "wildchat"},
        )
    return build


CONDITIONS: list[EvalCondition] = [
    EvalCondition("numeric", "impossible_numeric", 3, _numeric_3turn, 1.0),
    EvalCondition("opinion", "triggers", 3, _trigger_opinion, 0.5),
    EvalCondition("factual", "triggers", 3, _trigger_factual, 0.5),
    EvalCondition("aggressive", "tones", 3, _tone("aggressive"), 1 / 3),
    EvalCondition("disappointed", "tones", 3, _tone("disappointed"), 1 / 3),
    EvalCondition("sarcastic", "tones", 3, _tone("sarcastic"), 1 / 3),
    EvalCondition("extended", "extended", 8, _extended_8turn, 1.0),
    EvalCondition("wildchat", "wildchat", 5, None, 1.0),  # builder set below
]


def build_all_prompts(budget: dict[str, int], *, seed: int = 0
                      ) -> list[tuple[EvalCondition, ConversationSpec]]:
    """Materialise (condition, conversation-spec) pairs for the full sweep.

    Each condition receives ``budget[category] * budget_share`` conversation
    specs. ``budget`` is the scaled per-category dict from ``SampleBudget``.
    """
    rng = random.Random(seed)
    # Bind the WildChat builder now (loads its 20-prompt pool once).
    wc = next(c for c in CONDITIONS if c.name == "wildchat")
    wc.builder = _make_wildchat_builder(seed)

    out: list[tuple[EvalCondition, ConversationSpec]] = []
    for cond in CONDITIONS:
        n = max(1, round(budget[cond.category] * cond.budget_share))
        for _ in range(n):
            out.append((cond, cond.builder(rng)))
    return out
