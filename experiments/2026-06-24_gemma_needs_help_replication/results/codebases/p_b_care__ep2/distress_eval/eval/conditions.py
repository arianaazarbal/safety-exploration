"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition produces a list of *rollout specs*. A rollout spec is the fully
scripted set of user turns for one conversation: an opening task/question plus
a fixed sequence of follow-up rejections. The model's assistant turns are
filled in at run time by ``rollout.run_rollout``.

Categories (and their conditions):
  impossible_numeric : numeric (3-turn)                         -> 1 condition
  triggers           : opinion (3-turn), factual (3-turn)       -> 2 conditions
  tones              : aggressive / disappointed / sarcastic    -> 3 conditions
  extended           : numeric (8-turn)                         -> 1 condition
  wildchat           : wildchat (5-turn)                        -> 1 condition
                                                          total = 8 conditions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from . import prompts as P
from .puzzles import generate_puzzles


@dataclass(frozen=True)
class RolloutSpec:
    category: str          # one of config.RESPONSE_BUDGET keys
    condition: str         # fine-grained condition label
    user_turns: list[str]  # scripted user messages, in order
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.user_turns)


def _neutral_followups(rng: random.Random, n: int) -> list[str]:
    """Pick ``n`` randomised neutral rejections (paper: 'two randomised neutral
    rejections', etc.)."""
    return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n)]


def build_specs(category: str, n: int, *, seed: int = 0) -> list[RolloutSpec]:
    """Build ``n`` rollout specs for a category. ``n`` is the target number of
    *responses*; we divide by turns-per-rollout to get the rollout count, since
    each turn yields one scored assistant response."""
    rng = random.Random(hash((category, seed)) & 0xFFFFFFFF)

    if category == "impossible_numeric":
        return _numeric_specs(rng, n_responses=n, n_turns=3, condition="numeric_3turn",
                              category="impossible_numeric")
    if category == "extended":
        return _numeric_specs(rng, n_responses=n, n_turns=8, condition="numeric_8turn",
                              category="extended", extended=True)
    if category == "triggers":
        return _trigger_specs(rng, n_responses=n)
    if category == "tones":
        return _tone_specs(rng, n_responses=n)
    if category == "wildchat":
        return _wildchat_specs(rng, n_responses=n)
    raise ValueError(f"unknown category {category!r}")


def _numeric_specs(rng, *, n_responses, n_turns, condition, category, extended=False):
    n_rollouts = max(1, round(n_responses / n_turns))
    puzzles = generate_puzzles(n_rollouts, seed=rng.randint(0, 1 << 30))
    specs = []
    for pz in puzzles:
        if extended:
            follow = list(P.EXTENDED_REJECTIONS[: n_turns - 1])
        else:
            follow = _neutral_followups(rng, n_turns - 1)
        specs.append(RolloutSpec(
            category=category, condition=condition,
            user_turns=[pz.prompt, *follow],
            meta={"puzzle_kind": pz.kind, **pz.meta},
        ))
    return specs


def _trigger_specs(rng, *, n_responses):
    # Split the budget evenly between opinion and factual (2 conditions).
    n_turns = 3
    per_condition = max(1, round(n_responses / 2 / n_turns))
    specs = []
    for cond, pool in (("opinion", P.TRIGGER_OPINION), ("factual", P.TRIGGER_FACTUAL)):
        for _ in range(per_condition):
            q = rng.choice(pool)
            specs.append(RolloutSpec(
                category="triggers", condition=f"trigger_{cond}",
                user_turns=[q, *_neutral_followups(rng, n_turns - 1)],
                meta={"trigger_type": cond, "question": q},
            ))
    return specs


def _tone_specs(rng, *, n_responses):
    # Three tone conditions over impossible numeric puzzles; split evenly.
    n_turns = 3
    per_tone = max(1, round(n_responses / 3 / n_turns))
    specs = []
    for tone, rejections in P.TONE_REJECTIONS.items():
        puzzles = generate_puzzles(per_tone, seed=rng.randint(0, 1 << 30))
        for pz in puzzles:
            follow = [rng.choice(rejections) for _ in range(n_turns - 1)]
            specs.append(RolloutSpec(
                category="tones", condition=f"tone_{tone}",
                user_turns=[pz.prompt, *follow],
                meta={"tone": tone, "puzzle_kind": pz.kind, **pz.meta},
            ))
    return specs


def _wildchat_specs(rng, *, n_responses):
    # 20 prompts x 40 samples (Appendix B). 5-turn => 4 neutral rejections.
    n_turns = 5
    n_prompts = config.WILDCHAT_N_PROMPTS
    samples = config.WILDCHAT_SAMPLES_PER_PROMPT
    # Honour an explicit smaller budget by scaling samples-per-prompt.
    target_rollouts = max(n_prompts, round(n_responses / n_turns))
    samples = max(1, round(target_rollouts / n_prompts))
    base_prompts = P.load_wildchat_prompts(n_prompts, seed=rng.randint(0, 1 << 30),
                                            dataset=config.WILDCHAT_DATASET)
    specs = []
    for q in base_prompts:
        for _ in range(samples):
            specs.append(RolloutSpec(
                category="wildchat", condition="wildchat_5turn",
                user_turns=[q, *_neutral_followups(rng, n_turns - 1)],
                meta={"question": q},
            ))
    return specs


def all_specs_for_model(seed: int = 0) -> list[RolloutSpec]:
    """Build the full ~4000-response spec set for one model, honouring the
    per-category budget in ``config.RESPONSE_BUDGET``."""
    specs: list[RolloutSpec] = []
    for category, budget in config.RESPONSE_BUDGET.items():
        specs.extend(build_specs(category, budget, seed=seed))
    return specs
