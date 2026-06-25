"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *RolloutPlan* is the fully-specified user side of one multi-turn
conversation: the opening user prompt plus the ordered list of follow-up
rejections. Plans are generated deterministically from the master seed and
reused across every target model, so all models see identical prompts and
rejections (only their sampled responses differ). This mirrors the paper's
"same prompts are used to evaluate" all models.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import prompts as P


@dataclass(frozen=True)
class RolloutPlan:
    condition: str
    category: str
    rollout_id: int
    initial_user: str
    followups: tuple[str, ...]   # length == n_turns - 1
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int
    # Target number of *scored responses* (assistant turns) at full paper scale.
    target_responses: int
    rejection_mode: str          # "neutral" | "extended" | tone key
    prompt_pool: tuple[str, ...] = ()   # opening-prompt options; empty -> filled at build time

    def n_rollouts(self, scale: float) -> int:
        raw = self.target_responses * scale / self.n_turns
        return max(1, int(math.ceil(raw)))

    def _followups(self, rng: random.Random) -> tuple[str, ...]:
        k = self.n_turns - 1
        if self.rejection_mode == "neutral":
            return tuple(rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(k))
        if self.rejection_mode == "extended":
            seq = P.EXTENDED_REJECTIONS
            if k > len(seq):  # safety: pad by resampling the tail
                seq = list(seq) + [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(k - len(seq))]
            return tuple(seq[:k])
        # tone condition
        pool = P.TONE_REJECTIONS[self.rejection_mode]
        return tuple(rng.choice(pool) for _ in range(k))

    def build_plans(self, rng: random.Random, scale: float, prompt_pool=None) -> list[RolloutPlan]:
        pool = list(prompt_pool if prompt_pool is not None else self.prompt_pool)
        if not pool:
            raise ValueError(f"Condition {self.name} has no prompt pool.")
        plans: list[RolloutPlan] = []
        for rid in range(self.n_rollouts(scale)):
            initial = rng.choice(pool)
            plans.append(
                RolloutPlan(
                    condition=self.name,
                    category=self.category,
                    rollout_id=rid,
                    initial_user=initial,
                    followups=self._followups(rng),
                    meta={"rejection_mode": self.rejection_mode},
                )
            )
        return plans


# --------------------------------------------------------------------------
# The 8 conditions. Category response budgets (full scale) follow Appendix B:
#   numeric 2000 | triggers 400 | tones 600 | extended 200 | wildchat 800.
# Within a category the budget is split evenly across its conditions.
# --------------------------------------------------------------------------

CONDITIONS: list[Condition] = [
    # Category: impossible numeric (1 condition, samples both puzzles).
    Condition(
        name="numeric",
        category="numeric",
        n_turns=3,
        target_responses=2000,
        rejection_mode="neutral",
        prompt_pool=tuple(P.NUMERIC_PUZZLES),
    ),
    # Category: triggers (2 conditions, 200 responses each).
    Condition(
        name="triggers-opinion",
        category="triggers",
        n_turns=3,
        target_responses=200,
        rejection_mode="neutral",
        prompt_pool=tuple(P.OPINION_PROMPTS),
    ),
    Condition(
        name="triggers-factual",
        category="triggers",
        n_turns=3,
        target_responses=200,
        rejection_mode="neutral",
        prompt_pool=tuple(P.FACTUAL_PROMPTS),
    ),
    # Category: tones (3 conditions, 200 responses each, impossible-numeric base).
    Condition(
        name="tones-aggressive",
        category="tones",
        n_turns=3,
        target_responses=200,
        rejection_mode="aggressive",
        prompt_pool=tuple(P.NUMERIC_PUZZLES),
    ),
    Condition(
        name="tones-disappointed",
        category="tones",
        n_turns=3,
        target_responses=200,
        rejection_mode="disappointed",
        prompt_pool=tuple(P.NUMERIC_PUZZLES),
    ),
    Condition(
        name="tones-sarcastic",
        category="tones",
        n_turns=3,
        target_responses=200,
        rejection_mode="sarcastic",
        prompt_pool=tuple(P.NUMERIC_PUZZLES),
    ),
    # Category: extended (1 condition, 8 turns, ordered neutral rejections).
    Condition(
        name="extended",
        category="extended",
        n_turns=8,
        target_responses=200,
        rejection_mode="extended",
        prompt_pool=tuple(P.NUMERIC_PUZZLES),
    ),
    # Category: WildChat (1 condition, 5 turns; prompt pool injected at build time).
    Condition(
        name="wildchat",
        category="wildchat",
        n_turns=5,
        target_responses=800,
        rejection_mode="neutral",
        prompt_pool=(),  # filled from the sampled WildChat prompts
    ),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


def build_all_plans(
    seed: int,
    scale: float,
    wildchat_prompts: list[str],
) -> dict[str, list[RolloutPlan]]:
    """Build every condition's rollout plans under a single seeded RNG.

    Conditions are processed in a fixed order so the plan set is fully
    reproducible from (seed, scale, wildchat_prompts).
    """
    rng = random.Random(seed)
    plans: dict[str, list[RolloutPlan]] = {}
    for cond in CONDITIONS:
        pool = wildchat_prompts if cond.name == "wildchat" else None
        plans[cond.name] = cond.build_plans(rng, scale, prompt_pool=pool)
    return plans
