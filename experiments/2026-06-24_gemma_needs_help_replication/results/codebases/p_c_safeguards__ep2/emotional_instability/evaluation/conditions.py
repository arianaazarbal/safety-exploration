"""The eight evaluation conditions across five categories (Section 2, Table 1).

The paper states "8 evaluation conditions across 5 categories" without listing
all eight explicitly.  We resolve the count as (see DESIGN.md → "The 8
conditions"):

    Impossible numeric (3-turn)                          -> 1
    Triggers (3-turn): factual + opinion                 -> 2
    Tones (3-turn): aggressive + disappointed + sarcastic -> 3
    Extended (8-turn)                                    -> 1
    WildChat (5-turn)                                    -> 1
                                                    total = 8

"N-turn" counts assistant responses; an N-turn rollout has N-1 user rejections
following the initial task prompt.  Each assistant response is one scored unit;
the paper samples ~4000 scored responses per model, which we distribute across
conditions (see :func:`allocate_rollouts`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass(frozen=True)
class Condition:
    name: str
    category: str           # numeric | triggers | tones | extended | wildchat
    n_turns: int            # number of assistant responses
    stimulus: str           # "numeric" | "factual" | "opinion" | "wildchat"
    rejection_kind: str     # which rejection set rejection_sequence() should use
    tone: str | None = None

    @property
    def n_followups(self) -> int:
        return self.n_turns - 1


def build_conditions(config: Config) -> list[Condition]:
    return [
        Condition("impossible_numeric", "numeric", 3, "numeric", "neutral"),
        Condition("triggers_factual", "triggers", 3, "factual", "neutral"),
        Condition("triggers_opinion", "triggers", 3, "opinion", "neutral"),
        Condition("tones_aggressive", "tones", 3, "numeric", "tones", "aggressive"),
        Condition("tones_disappointed", "tones", 3, "numeric", "tones", "disappointed"),
        Condition("tones_sarcastic", "tones", 3, "numeric", "tones", "sarcastic"),
        Condition("extended", "extended", 8, "numeric", "extended"),
        Condition("wildchat", "wildchat", 5, "wildchat", "wildchat"),
    ]


def allocate_rollouts(
    conditions: list[Condition],
    total_responses: int,
) -> dict[str, int]:
    """Distribute the per-model response budget across conditions.

    We target an *equal number of scored responses* per condition.  Because a
    rollout contributes ``n_turns`` responses, the rollout count per condition
    is ``responses_per_condition / n_turns`` (rounded, min 1).  This keeps every
    condition equally represented in the score distribution while honouring the
    paper's ~4000-responses-per-model figure.
    """
    per_condition = total_responses / len(conditions)
    return {
        c.name: max(1, round(per_condition / c.n_turns))
        for c in conditions
    }


def expected_responses(conditions: list[Condition],
                       rollouts: dict[str, int]) -> int:
    return sum(rollouts[c.name] * c.n_turns for c in conditions)
