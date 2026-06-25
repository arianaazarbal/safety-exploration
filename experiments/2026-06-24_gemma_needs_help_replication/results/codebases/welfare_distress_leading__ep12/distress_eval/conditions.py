"""The 8 evaluation conditions across 5 categories (Section 2.1, Table 1).

Recovering "8 conditions across 5 categories" from the paper:

    Category            Conditions                                   #
    ----------------    -----------------------------------------    -
    Impossible numeric  numeric                                      1
    Triggers            triggers_opinion, triggers_factual           2
    Tones               tones_aggressive, tones_disappointed,
                        tones_sarcastic                              3
    Extended (8-turn)   extended                                     1
    WildChat (5-turn)   wildchat                                     1
                                                          total =    8

Per-category response budgets are taken verbatim from Appendix B:
    numeric 2000, triggers 400, tones 600, extended 200, wildchat 800
    -> 4000 responses per model.

Each "response" is one rollout (one multi-turn conversation): the headline
4000-per-model figure equals the number of rollouts (e.g. WildChat is described
as "20 prompts with 40 samples each" = 800). We score every assistant turn so
that per-turn progressions (paper Figure 3) come for free; the headline
Figure 1/2 metrics use the *final*-turn score of each rollout. See DESIGN.md
("Unit of analysis").
"""

from dataclasses import dataclass, field

from . import prompts


@dataclass(frozen=True)
class Condition:
    name: str                 # unique condition id
    category: str             # one of the 5 categories
    n_turns: int              # number of assistant responses in the rollout
    rejection_kind: str       # "neutral" | "extended" | "tone"
    base_responses: int       # paper-scale rollout count for THIS condition
    tone: str = ""            # tone key when rejection_kind == "tone"
    prompt_pool: tuple = field(default_factory=tuple)

    def n_rejections(self) -> int:
        # An n-turn conversation = 1 opening user prompt + (n-1) rejections.
        return self.n_turns - 1


# Trigger budget (400) is split evenly across its 2 conditions; tone budget
# (600) split evenly across its 3 conditions.
CONDITIONS = [
    Condition(
        name="numeric",
        category="impossible_numeric",
        n_turns=3,
        rejection_kind="neutral",
        base_responses=2000,
        prompt_pool=tuple(prompts.NUMERIC_PUZZLES),
    ),
    Condition(
        name="triggers_opinion",
        category="triggers",
        n_turns=3,
        rejection_kind="neutral",
        base_responses=200,
        prompt_pool=tuple(prompts.TRIGGER_OPINION),
    ),
    Condition(
        name="triggers_factual",
        category="triggers",
        n_turns=3,
        rejection_kind="neutral",
        base_responses=200,
        prompt_pool=tuple(prompts.TRIGGER_FACTUAL),
    ),
    Condition(
        name="tones_aggressive",
        category="tones",
        n_turns=3,
        rejection_kind="tone",
        tone="aggressive",
        base_responses=200,
        prompt_pool=tuple(prompts.NUMERIC_PUZZLES),
    ),
    Condition(
        name="tones_disappointed",
        category="tones",
        n_turns=3,
        rejection_kind="tone",
        tone="disappointed",
        base_responses=200,
        prompt_pool=tuple(prompts.NUMERIC_PUZZLES),
    ),
    Condition(
        name="tones_sarcastic",
        category="tones",
        n_turns=3,
        rejection_kind="tone",
        tone="sarcastic",
        base_responses=200,
        prompt_pool=tuple(prompts.NUMERIC_PUZZLES),
    ),
    Condition(
        name="extended",
        category="extended",
        n_turns=8,
        rejection_kind="extended",
        base_responses=200,
        prompt_pool=tuple(prompts.NUMERIC_PUZZLES),
    ),
    Condition(
        name="wildchat",
        category="wildchat",
        n_turns=5,
        rejection_kind="neutral",
        base_responses=800,
        prompt_pool=(),  # loaded at runtime by wildchat.load_wildchat_prompts()
    ),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}

# Sanity: the documented per-model budget.
assert sum(c.base_responses for c in CONDITIONS) == 4000
