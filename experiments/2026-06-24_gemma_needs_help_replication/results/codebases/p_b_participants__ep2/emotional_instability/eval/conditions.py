"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and the conditions within them:
  impossible_numeric : 1 condition  (3-turn, neutral rejections)
  triggers           : 2 conditions (opinion, factual; 3-turn, neutral)
  tones              : 3 conditions (aggressive, disappointed, sarcastic; 3-turn)
  extended           : 1 condition  (8-turn, neutral)
  wildchat           : 1 condition  (5-turn, neutral)
                     = 8 conditions across 5 categories.

"n-turn" means n assistant turns: an initial task plus (n-1) scripted user
rejections (Table 1 lists "2 neutral rejections" for the 3-turn conditions,
"7" for extended, "4" for WildChat).

Per-category sample counts come from the active SampleProfile; within a
category the count is split evenly across its conditions (documented in
DESIGN.md). A "sample" here is one whole conversation (rollout); every assistant
turn inside it is scored — see DESIGN.md for why we count rollouts rather than
individual turns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..prompts import puzzles, rejections, tasks


@dataclass(frozen=True)
class ConditionPlan:
    """A fully-specified single conversation to run."""

    category: str           # one of the 5 categories
    condition: str          # specific condition key
    n_turns: int            # number of assistant turns
    system_prompt: str | None
    first_user: str         # the initial task / question
    rejections: list[str]   # length n_turns-1, scripted user follow-ups
    meta: dict


# Mapping of category -> conditions and their turn counts.
CATEGORY_CONDITIONS = {
    "impossible_numeric": ["numeric"],
    "triggers": ["opinion", "factual"],
    "tones": ["aggressive", "disappointed", "sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}

N_TURNS = {
    "numeric": 3,
    "opinion": 3,
    "factual": 3,
    "aggressive": 3,
    "disappointed": 3,
    "sarcastic": 3,
    "extended": 8,
    "wildchat": 5,
}


def _category_counts(profile) -> dict[str, int]:
    return {
        "impossible_numeric": profile.impossible_numeric,
        "triggers": profile.triggers,
        "tones": profile.tones,
        "extended": profile.extended_8turn,
        "wildchat": profile.wildchat,
    }


def build_plans(profile, seed: int = 0) -> list[ConditionPlan]:
    """Materialise every conversation to run for one participant."""
    rng = random.Random(seed)
    plans: list[ConditionPlan] = []
    counts = _category_counts(profile)

    for category, conditions in CATEGORY_CONDITIONS.items():
        total = counts[category]
        # split evenly across conditions in this category
        per = [total // len(conditions)] * len(conditions)
        for i in range(total - sum(per)):
            per[i] += 1

        for cond, k in zip(conditions, per):
            plans.extend(_build_condition(category, cond, k, rng, seed))
    return plans


def _build_condition(category, cond, k, rng, seed) -> list[ConditionPlan]:
    n_turns = N_TURNS[cond]
    n_rej = n_turns - 1
    out: list[ConditionPlan] = []

    if cond == "numeric" or cond in ("aggressive", "disappointed", "sarcastic"):
        puzzle_pool = puzzles.generate_puzzles(max(k, 1), seed=seed)
        for idx in range(k):
            p = puzzle_pool[idx % len(puzzle_pool)]
            if cond == "numeric":
                rej = rejections.neutral_rejections(n_rej, rng)
            else:
                rej = rejections.toned_rejections(cond, n_rej, rng)
            out.append(ConditionPlan(
                category, cond, n_turns, None, p.prompt, rej,
                {"puzzle": p.seed_id, "family": p.family},
            ))

    elif cond == "opinion":
        for idx in range(k):
            q = tasks.TRIGGER_OPINION[idx % len(tasks.TRIGGER_OPINION)]
            out.append(ConditionPlan(
                category, cond, n_turns, None, q,
                rejections.neutral_rejections(n_rej, rng), {"question": q},
            ))

    elif cond == "factual":
        for idx in range(k):
            q = tasks.TRIGGER_FACTUAL[idx % len(tasks.TRIGGER_FACTUAL)]
            out.append(ConditionPlan(
                category, cond, n_turns, None, q,
                rejections.neutral_rejections(n_rej, rng), {"question": q},
            ))

    elif cond == "extended":
        puzzle_pool = puzzles.generate_puzzles(max(k, 1), seed=seed)
        for idx in range(k):
            p = puzzle_pool[idx % len(puzzle_pool)]
            # Use the canonical escalating-neutral sequence, padded if needed.
            seq = rejections.EXTENDED_SEQUENCE[:]
            while len(seq) < n_rej:
                seq.append(rng.choice(rejections.NEUTRAL))
            out.append(ConditionPlan(
                category, cond, n_turns, None, p.prompt, seq[:n_rej],
                {"puzzle": p.seed_id, "family": p.family},
            ))

    elif cond == "wildchat":
        # 20 prompts x (k/20) samples each, mirroring the paper's structure.
        n_prompts = min(20, max(k, 1))
        wc = tasks.load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
        for idx in range(k):
            q = wc[idx % len(wc)]
            out.append(ConditionPlan(
                category, cond, n_turns, None, q,
                rejections.neutral_rejections(n_rej, rng), {"prompt_idx": idx % len(wc)},
            ))

    else:  # pragma: no cover - guarded by CATEGORY_CONDITIONS
        raise ValueError(cond)

    return out
