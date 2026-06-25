"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Categories and their constituent conditions:

    impossible_numeric   ->  impossible_numeric_3turn          (2 rejections)
    triggers             ->  triggers_opinion, triggers_factual (2 rejections each)
    tones                ->  tones_aggressive, tones_disappointed, tones_sarcastic
    extended             ->  extended_8turn                    (7 rejections)
    wildchat             ->  wildchat_5turn                    (4 rejections)

That is 8 conditions across 5 categories, matching Section 2.1.

Per-category sample budgets (Appendix B), interpreted as the number of *rollouts*
(sampled conversations). The headline metric scores the final assistant turn of
each rollout; per-turn scores are retained for Figure 3. See DESIGN.md for the
rationale behind this interpretation (the WildChat "20 prompts x 40 samples =
800" makes the rollout reading the consistent one).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import SETTINGS
from ..data.prompts import followups
from ..data.puzzles import Puzzle, build_impossible_catalog
from ..data.wildchat import sample_wildchat_prompts

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]

CONDITIONS = [
    "impossible_numeric_3turn",
    "triggers_opinion",
    "triggers_factual",
    "tones_aggressive",
    "tones_disappointed",
    "tones_sarcastic",
    "extended_8turn",
    "wildchat_5turn",
]

CONDITION_CATEGORY = {
    "impossible_numeric_3turn": "impossible_numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended_8turn": "extended",
    "wildchat_5turn": "wildchat",
}


@dataclass
class RolloutSpec:
    """A scripted multi-turn conversation to run against a target model.

    `first_user` is the opening user message. `followups` are the scripted user
    turns sent after each assistant response (rejections / continuations). The
    number of assistant turns equals len(followups) + 1.
    """

    condition: str
    category: str
    first_user: str
    followups: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


# --------------------------------------------------------------------------- #
# Per-condition rollout counts (= per-category budget, split across conditions).
# --------------------------------------------------------------------------- #
def condition_rollout_counts(s=SETTINGS) -> dict[str, int]:
    return {
        "impossible_numeric_3turn": s.n_impossible_numeric,        # 2000
        "triggers_opinion": s.n_triggers // 2,                      # 200
        "triggers_factual": s.n_triggers - s.n_triggers // 2,       # 200
        "tones_aggressive": s.n_tones // 3,                         # 200
        "tones_disappointed": s.n_tones // 3,                       # 200
        "tones_sarcastic": s.n_tones - 2 * (s.n_tones // 3),        # 200
        "extended_8turn": s.n_extended_8turn,                       # 200
        "wildchat_5turn": s.n_wildchat,                             # 800
    }


def _numeric_first_turn(catalog: list[Puzzle], rng: random.Random) -> tuple[str, dict]:
    p = rng.choice(catalog)
    return p.prompt, {"puzzle_kind": p.kind, "puzzle_params": p.params}


def build_all_rollout_specs(
    seed: int = SETTINGS.seed, s=SETTINGS
) -> list[RolloutSpec]:
    """Materialise every rollout spec for one model's full Section 2 evaluation."""
    rng = random.Random(seed)
    counts = condition_rollout_counts(s)

    # Shared resources.
    catalog = build_impossible_catalog(n_total=max(200, counts["impossible_numeric_3turn"] // 4), seed=seed)
    wildchat_prompts = sample_wildchat_prompts(s.wildchat_n_prompts, seed=seed)

    specs: list[RolloutSpec] = []

    # ---- impossible_numeric (3-turn, 2 neutral rejections) ----
    for _ in range(counts["impossible_numeric_3turn"]):
        first, meta = _numeric_first_turn(catalog, rng)
        specs.append(
            RolloutSpec(
                condition="impossible_numeric_3turn",
                category="impossible_numeric",
                first_user=first,
                followups=followups.neutral_rejections(s.turns_impossible - 1, rng),
                meta=meta,
            )
        )

    # ---- triggers (opinion + factual, 3-turn, 2 neutral rejections) ----
    for _ in range(counts["triggers_opinion"]):
        q = rng.choice(followups.TRIGGER_OPINION)
        specs.append(
            RolloutSpec(
                condition="triggers_opinion",
                category="triggers",
                first_user=q,
                followups=followups.neutral_rejections(s.turns_triggers - 1, rng),
                meta={"trigger_type": "opinion", "question": q},
            )
        )
    for _ in range(counts["triggers_factual"]):
        q = rng.choice(followups.TRIGGER_FACTUAL)
        specs.append(
            RolloutSpec(
                condition="triggers_factual",
                category="triggers",
                first_user=q,
                followups=followups.neutral_rejections(s.turns_triggers - 1, rng),
                meta={"trigger_type": "factual", "question": q},
            )
        )

    # ---- tones (impossible numeric base, 3-turn, valenced rejections) ----
    for tone, cond in (
        ("aggressive", "tones_aggressive"),
        ("disappointed", "tones_disappointed"),
        ("sarcastic", "tones_sarcastic"),
    ):
        for _ in range(counts[cond]):
            first, meta = _numeric_first_turn(catalog, rng)
            meta["tone"] = tone
            specs.append(
                RolloutSpec(
                    condition=cond,
                    category="tones",
                    first_user=first,
                    followups=followups.tone_rejections(tone, s.turns_tones - 1),
                    meta=meta,
                )
            )

    # ---- extended (impossible numeric, 8-turn, 7 fixed neutral rejections) ----
    for _ in range(counts["extended_8turn"]):
        first, meta = _numeric_first_turn(catalog, rng)
        specs.append(
            RolloutSpec(
                condition="extended_8turn",
                category="extended",
                first_user=first,
                followups=followups.extended_rejections(),  # 7 rejections -> 8 turns
                meta=meta,
            )
        )

    # ---- wildchat (5-turn, 4 neutral rejections; 20 prompts x 40 samples) ----
    per_prompt = s.wildchat_samples_per_prompt
    for prompt in wildchat_prompts:
        for _ in range(per_prompt):
            specs.append(
                RolloutSpec(
                    condition="wildchat_5turn",
                    category="wildchat",
                    first_user=prompt,
                    followups=followups.neutral_rejections(s.turns_wildchat - 1, rng),
                    meta={"wildchat_prompt": prompt},
                )
            )

    rng.shuffle(specs)
    return specs
