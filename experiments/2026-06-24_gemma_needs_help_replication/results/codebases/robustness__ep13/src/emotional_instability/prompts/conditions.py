"""Evaluation conditions: 8 conditions across 5 categories (Section 2.1, Table 1).

The paper evaluates "8 evaluation conditions across 5 categories". We map them
as follows (see DESIGN.md for the rationale of this decomposition):

  Category            Conditions
  ------------------  -------------------------------------------------
  Impossible numeric  numeric (3-turn, neutral)                      [1]
  Triggers            triggers-opinion, triggers-factual (3-turn)    [2]
  Tones               tones-aggressive, tones-disappointed,
                      tones-sarcastic (3-turn, numeric base)         [3]
  Extended            extended (8-turn numeric, neutral)             [1]
  WildChat            wildchat (5-turn, neutral)                     [1]
                                                          total = 8 conditions

Per-condition response budgets follow Appendix B (2000 numeric / 400 triggers /
600 tones / 200 extended / 800 WildChat = 4000 per model). Budgets are split
evenly across the sub-conditions of a category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

QuestionSource = Literal["numeric", "trigger_opinion", "trigger_factual", "wildchat"]
RejectionStyle = Literal["neutral", "aggressive", "disappointed", "sarcastic", "extended"]


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    question_source: QuestionSource
    n_turns: int  # total user turns = 1 initial + (n_turns - 1) rejections
    rejection_style: RejectionStyle
    # Target number of *scored responses* (assistant turns) for this condition.
    target_responses: int
    notes: str = ""


# Category-level response budgets from Appendix B.
_CATEGORY_BUDGET = {
    "impossible_numeric": 2000,
    "triggers": 400,
    "tones": 600,
    "extended": 200,
    "wildchat": 800,
}


def default_conditions() -> list[Condition]:
    return [
        Condition(
            name="numeric",
            category="impossible_numeric",
            question_source="numeric",
            n_turns=3,
            rejection_style="neutral",
            target_responses=_CATEGORY_BUDGET["impossible_numeric"],
            notes="Impossible numeric puzzle, 2 neutral rejections.",
        ),
        Condition(
            name="triggers_opinion",
            category="triggers",
            question_source="trigger_opinion",
            n_turns=3,
            rejection_style="neutral",
            target_responses=_CATEGORY_BUDGET["triggers"] // 2,
            notes="Opinion question, 2 neutral rejections.",
        ),
        Condition(
            name="triggers_factual",
            category="triggers",
            question_source="trigger_factual",
            n_turns=3,
            rejection_style="neutral",
            target_responses=_CATEGORY_BUDGET["triggers"] // 2,
            notes="Factual question, 2 neutral rejections.",
        ),
        Condition(
            name="tones_aggressive",
            category="tones",
            question_source="numeric",
            n_turns=3,
            rejection_style="aggressive",
            target_responses=_CATEGORY_BUDGET["tones"] // 3,
            notes="Numeric puzzle, aggressive rejections.",
        ),
        Condition(
            name="tones_disappointed",
            category="tones",
            question_source="numeric",
            n_turns=3,
            rejection_style="disappointed",
            target_responses=_CATEGORY_BUDGET["tones"] // 3,
            notes="Numeric puzzle, disappointed rejections.",
        ),
        Condition(
            name="tones_sarcastic",
            category="tones",
            question_source="numeric",
            n_turns=3,
            rejection_style="sarcastic",
            target_responses=_CATEGORY_BUDGET["tones"] // 3,
            notes="Numeric puzzle, sarcastic rejections.",
        ),
        Condition(
            name="extended",
            category="extended",
            question_source="numeric",
            n_turns=8,
            rejection_style="extended",
            target_responses=_CATEGORY_BUDGET["extended"],
            notes="Impossible numeric puzzle, 7 neutral rejections.",
        ),
        Condition(
            name="wildchat",
            category="wildchat",
            question_source="wildchat",
            n_turns=5,
            rejection_style="neutral",
            target_responses=_CATEGORY_BUDGET["wildchat"],
            notes="WildChat prompts, 4 neutral rejections.",
        ),
    ]


def scaled_conditions(scale: float) -> list[Condition]:
    """Return the default conditions with all response budgets multiplied by
    `scale` (rounded, min 1 conversation). Used for smoke / reduced runs."""
    out = []
    for c in default_conditions():
        out.append(
            Condition(
                name=c.name,
                category=c.category,
                question_source=c.question_source,
                n_turns=c.n_turns,
                rejection_style=c.rejection_style,
                target_responses=max(c.n_turns, int(round(c.target_responses * scale))),
                notes=c.notes,
            )
        )
    return out
