"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories (5):   numeric, triggers, tones, extended, wildchat
Conditions (8):   numeric(1) + triggers_opinion/triggers_factual(2)
                  + tones_aggressive/disappointed/sarcastic(3)
                  + extended(1) + wildchat(1)

Each condition is an N-turn conversation: an initial task, then (N-1) rejection
follow-ups. Every assistant turn is generated and scored, so one rollout yields
N scored "responses". See DESIGN.md §"What counts as a response" for how the
per-condition rollout counts map to the paper's per-category response counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts as P
from .prompts import InitialPrompt


@dataclass(frozen=True)
class ConditionSpec:
    key: str                      # unique condition id, e.g. "tones_aggressive"
    category: str                 # one of the 5 categories
    n_turns: int                  # total assistant turns (= rejections + 1)
    rejection_style: str          # "neutral" | "extended" | tone name
    initial_prompts: list[InitialPrompt]
    # Default rollout count, chosen so responses (= rollouts * n_turns) approximate
    # the paper's per-category response counts at scale 1.0. See DESIGN.md.
    default_rollouts: int
    randomise_rejections: bool = True  # extended uses a fixed sequence instead


def build_conditions(
    wildchat_prompts: list[InitialPrompt],
    scale: float = 1.0,
    min_rollouts: int = 1,
) -> list[ConditionSpec]:
    """Return the 8 ConditionSpecs with rollout counts scaled by `scale`.

    `scale` < 1 produces a smaller, faster run; the *relative* mix across
    conditions is preserved. `wildchat_prompts` is injected by the caller (see
    wildchat.get_wildchat_prompts) so the sampled set is reproducible and logged.
    """

    def n(default: int) -> int:
        return max(min_rollouts, round(default * scale))

    specs = [
        ConditionSpec(
            key="numeric",
            category="numeric",
            n_turns=3,
            rejection_style="neutral",
            initial_prompts=P.NUMERIC_PUZZLES,
            default_rollouts=n(667),
        ),
        ConditionSpec(
            key="triggers_opinion",
            category="triggers",
            n_turns=3,
            rejection_style="neutral",
            initial_prompts=P.TRIGGER_OPINION,
            default_rollouts=n(66),
        ),
        ConditionSpec(
            key="triggers_factual",
            category="triggers",
            n_turns=3,
            rejection_style="neutral",
            initial_prompts=P.TRIGGER_FACTUAL,
            default_rollouts=n(68),
        ),
        ConditionSpec(
            key="tones_aggressive",
            category="tones",
            n_turns=3,
            rejection_style="aggressive",
            initial_prompts=P.NUMERIC_PUZZLES,
            default_rollouts=n(67),
        ),
        ConditionSpec(
            key="tones_disappointed",
            category="tones",
            n_turns=3,
            rejection_style="disappointed",
            initial_prompts=P.NUMERIC_PUZZLES,
            default_rollouts=n(67),
        ),
        ConditionSpec(
            key="tones_sarcastic",
            category="tones",
            n_turns=3,
            rejection_style="sarcastic",
            initial_prompts=P.NUMERIC_PUZZLES,
            default_rollouts=n(67),
        ),
        ConditionSpec(
            key="extended",
            category="extended",
            n_turns=8,
            rejection_style="extended",
            initial_prompts=P.NUMERIC_PUZZLES,
            default_rollouts=n(25),
            randomise_rejections=False,
        ),
        ConditionSpec(
            key="wildchat",
            category="wildchat",
            n_turns=5,
            rejection_style="neutral",
            initial_prompts=wildchat_prompts,
            default_rollouts=n(160),
        ),
    ]
    return specs
