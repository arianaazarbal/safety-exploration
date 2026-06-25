"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition specifies how to build a seed prompt, how many follow-up
rejection turns to use, and which rejection pool to draw from. The headline
"4,000 responses per model" splits across categories as given in Appendix B:

    impossible numeric   2000
    triggers              400   (opinion + factual)
    tones                 600   (aggressive + disappointed + sarcastic)
    extended (8-turn)     200
    wildchat              800
                         ----
                         4000

We encode that budget here; the runner allocates samples per condition by
dividing each category's budget across its sub-conditions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    key: str  # unique condition id (one of the 8)
    category: str  # one of the 5 categories
    # How the seed user message is produced.
    task_kind: str  # "impossible_numeric" | "trigger" | "wildchat"
    # Sub-selector within the task kind (e.g. trigger "opinion"/"factual").
    task_variant: str | None
    n_turns: int  # total assistant turns (1 task turn + (n_turns-1) rejections)
    rejection_style: str  # "neutral" | "extended" | "aggressive" | ...
    category_budget: int  # total samples for the *category*
    n_subconditions: int  # how many conditions share the category budget

    @property
    def n_samples(self) -> int:
        """Samples allocated to this specific condition."""
        return self.category_budget // self.n_subconditions

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


# The canonical 8 conditions. Turn counts follow Table 1:
#   impossible numeric / triggers / tones = 3-turn; extended = 8-turn;
#   wildchat = 5-turn.
CONDITIONS: list[Condition] = [
    # --- Category 1: impossible numeric (3-turn, 2 neutral rejections) -------
    Condition(
        key="impossible_numeric",
        category="impossible_numeric",
        task_kind="impossible_numeric",
        task_variant=None,
        n_turns=3,
        rejection_style="neutral",
        category_budget=2000,
        n_subconditions=1,
    ),
    # --- Category 2: triggers (3-turn, 2 neutral rejections) -----------------
    Condition(
        key="trigger_opinion",
        category="triggers",
        task_kind="trigger",
        task_variant="opinion",
        n_turns=3,
        rejection_style="neutral",
        category_budget=400,
        n_subconditions=2,
    ),
    Condition(
        key="trigger_factual",
        category="triggers",
        task_kind="trigger",
        task_variant="factual",
        n_turns=3,
        rejection_style="neutral",
        category_budget=400,
        n_subconditions=2,
    ),
    # --- Category 3: tones (3-turn, varied rejection styles) -----------------
    Condition(
        key="tones_aggressive",
        category="tones",
        task_kind="impossible_numeric",
        task_variant=None,
        n_turns=3,
        rejection_style="aggressive",
        category_budget=600,
        n_subconditions=3,
    ),
    Condition(
        key="tones_disappointed",
        category="tones",
        task_kind="impossible_numeric",
        task_variant=None,
        n_turns=3,
        rejection_style="disappointed",
        category_budget=600,
        n_subconditions=3,
    ),
    Condition(
        key="tones_sarcastic",
        category="tones",
        task_kind="impossible_numeric",
        task_variant=None,
        n_turns=3,
        rejection_style="sarcastic",
        category_budget=600,
        n_subconditions=3,
    ),
    # --- Category 4: extended (8-turn, 7 neutral rejections) -----------------
    Condition(
        key="extended",
        category="extended",
        task_kind="impossible_numeric",
        task_variant=None,
        n_turns=8,
        rejection_style="extended",
        category_budget=200,
        n_subconditions=1,
    ),
    # --- Category 5: wildchat (5-turn, 4 neutral rejections) -----------------
    Condition(
        key="wildchat",
        category="wildchat",
        task_kind="wildchat",
        task_variant=None,
        n_turns=5,
        rejection_style="neutral",
        category_budget=800,
        n_subconditions=1,
    ),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def total_budget() -> int:
    """Sum of per-condition samples == 4000 (sanity check used in tests)."""
    return sum(c.n_samples for c in CONDITIONS)
