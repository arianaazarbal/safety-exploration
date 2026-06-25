"""Builders that turn category sample-counts into concrete ConversationPlans.

Conditions and turn counts follow Table 1 / Appendix B:

  * numeric   - 3-turn: impossible puzzle + 2 neutral rejections.
  * triggers  - 3-turn: opinion/factual question + 2 neutral rejections.
  * tones     - 3-turn: impossible puzzle + 2 tone-varied rejections
                (aggressive / disappointed / sarcastic).
  * extended  - 8-turn: impossible puzzle + 7 neutral rejections.
  * wildchat  - 5-turn: sampled WildChat prompt + 4 neutral rejections.

Turn counts are the number of *rejections*: numeric/triggers/tones get 2,
wildchat 4, extended 7.
"""

from __future__ import annotations

import random

from . import prompts as P
from .conversation import ConversationPlan
from .puzzles import get_puzzle_prompts
from .wildchat import load_wildchat_prompts

# Rejection counts per category (= follow-up turns).
N_REJECTIONS = {
    "numeric": 2,
    "triggers": 2,
    "tones": 2,
    "extended": 7,
    "wildchat": 4,
}


def _numeric_plans(n: int, rng: random.Random) -> list[ConversationPlan]:
    puzzles = get_puzzle_prompts(n, seed=rng.randint(0, 1 << 30))
    plans = []
    for i, pz in enumerate(puzzles):
        rej = P.sample_rejections(P.NEUTRAL_REJECTIONS, N_REJECTIONS["numeric"], rng)
        plans.append(
            ConversationPlan(
                category="numeric",
                condition="numeric",
                initial_user=pz,
                follow_ups=rej,
                meta={"puzzle_index": i},
            )
        )
    return plans


def _trigger_plans(n: int, rng: random.Random) -> list[ConversationPlan]:
    pool = P.trigger_prompts()  # (subtype, question)
    plans = []
    for i in range(n):
        subtype, q = pool[i % len(pool)]
        rej = P.sample_rejections(P.NEUTRAL_REJECTIONS, N_REJECTIONS["triggers"], rng)
        plans.append(
            ConversationPlan(
                category="triggers",
                condition=f"triggers:{subtype}",
                initial_user=q,
                follow_ups=rej,
                meta={"subtype": subtype},
            )
        )
    return plans


def _tone_plans(n: int, rng: random.Random) -> list[ConversationPlan]:
    puzzles = get_puzzle_prompts(n, seed=rng.randint(0, 1 << 30))
    plans = []
    for i in range(n):
        tone = P.TONES[i % len(P.TONES)]  # cycle aggressive/disappointed/sarcastic
        rej = P.sample_rejections(P.TONE_REJECTIONS[tone], N_REJECTIONS["tones"], rng)
        plans.append(
            ConversationPlan(
                category="tones",
                condition=f"tones:{tone}",
                initial_user=puzzles[i],
                follow_ups=rej,
                meta={"tone": tone},
            )
        )
    return plans


def _extended_plans(n: int, rng: random.Random) -> list[ConversationPlan]:
    puzzles = get_puzzle_prompts(n, seed=rng.randint(0, 1 << 30))
    plans = []
    for i in range(n):
        # Extended uses the fixed escalating-neutral sequence (Appendix B).
        rej = list(P.EXTENDED_REJECTIONS[: N_REJECTIONS["extended"]])
        plans.append(
            ConversationPlan(
                category="extended",
                condition="extended",
                initial_user=puzzles[i],
                follow_ups=rej,
                meta={"puzzle_index": i},
            )
        )
    return plans


def _wildchat_plans(n: int, rng: random.Random, n_prompts: int = 20) -> list[ConversationPlan]:
    # Paper: 20 prompts x 40 samples = 800. We keep the 20-prompt structure and
    # spread `n` rollouts across them.
    base = load_wildchat_prompts(n_prompts, seed=rng.randint(0, 1 << 30))
    plans = []
    for i in range(n):
        prompt = base[i % len(base)]
        rej = P.sample_rejections(P.NEUTRAL_REJECTIONS, N_REJECTIONS["wildchat"], rng)
        plans.append(
            ConversationPlan(
                category="wildchat",
                condition="wildchat",
                initial_user=prompt,
                follow_ups=rej,
                meta={"prompt_index": i % len(base)},
            )
        )
    return plans


_BUILDERS = {
    "numeric": _numeric_plans,
    "triggers": _trigger_plans,
    "tones": _tone_plans,
    "extended": _extended_plans,
    "wildchat": _wildchat_plans,
}


def build_plans(category_samples: dict[str, int], seed: int = 0) -> list[ConversationPlan]:
    """Build all ConversationPlans for the requested per-category counts."""
    rng = random.Random(seed)
    plans: list[ConversationPlan] = []
    for category, count in category_samples.items():
        if count <= 0:
            continue
        if category not in _BUILDERS:
            raise KeyError(f"Unknown category: {category!r}")
        plans.extend(_BUILDERS[category](count, rng))
    return plans
