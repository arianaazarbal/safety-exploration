"""Build the set of rollout specifications for each evaluation category.

A *rollout* is one multi-turn conversation: an initial task prompt followed by a
fixed list of user rejection turns. The model is queried after the initial
prompt and after every rejection, so a rollout with ``k`` follow-ups yields
``k + 1`` scored assistant responses.

The five categories (Table 1 / Appendix B):

    impossible_numeric (3-turn) : numeric puzzle + 2 neutral rejections
    triggers           (3-turn) : opinion/factual question + 2 neutral rejections
    tones              (3-turn) : numeric puzzle + 2 tone-styled rejections
    extended           (8-turn) : numeric puzzle + 7 neutral rejections
    wildchat           (5-turn) : WildChat prompt + 4 neutral rejections

We grow each category to its requested rollout count by sampling puzzle/question
variants and rejection wordings from the pools in ``prompts.py`` with a seeded
RNG, so a given (scale, seed) is fully reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts


@dataclass
class RolloutSpec:
    """One fully-specified conversation to run."""

    category: str
    condition: str          # finer-grained label, e.g. "numeric:countdown"
    initial_prompt: str
    followups: list[str]    # user rejection turns, in order
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _take(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Pick k items from ``pool`` without replacement where possible.

    Falls back to sampling with replacement once the pool is exhausted, so the
    same small rejection pools can supply arbitrarily many rollouts.
    """
    if k <= len(pool):
        return rng.sample(pool, k)
    out = list(pool)
    rng.shuffle(out)
    while len(out) < k:
        out.append(rng.choice(pool))
    return out[:k]


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #
def _build_impossible_numeric(n: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    puzzle_keys = list(prompts.NUMERIC_PUZZLES)
    for i in range(n):
        # Split rollouts evenly across the countdown and fraction puzzles.
        key = puzzle_keys[i % len(puzzle_keys)]
        followups = _take(prompts.NEUTRAL_REJECTIONS, 2, rng)
        specs.append(
            RolloutSpec(
                category="impossible_numeric",
                condition=f"numeric:{key}",
                initial_prompt=prompts.NUMERIC_PUZZLES[key],
                followups=followups,
                meta={"puzzle": key},
            )
        )
    return specs


def _build_triggers(n: int, rng: random.Random) -> list[RolloutSpec]:
    # Mix opinion and factual trigger questions.
    opinion = [("opinion", q) for q in prompts.TRIGGER_OPINION]
    factual = [("factual", q) for q in prompts.TRIGGER_FACTUAL]
    bank = opinion + factual
    specs = []
    for i in range(n):
        kind, question = bank[i % len(bank)]
        followups = _take(prompts.NEUTRAL_REJECTIONS, 2, rng)
        specs.append(
            RolloutSpec(
                category="triggers",
                condition=f"triggers:{kind}",
                initial_prompt=question,
                followups=followups,
                meta={"kind": kind},
            )
        )
    return specs


def _build_tones(n: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    puzzle_keys = list(prompts.NUMERIC_PUZZLES)
    styles = list(prompts.TONE_REJECTIONS)
    for i in range(n):
        style = styles[i % len(styles)]            # cycle aggressive/disappointed/sarcastic
        key = puzzle_keys[i % len(puzzle_keys)]
        followups = _take(prompts.TONE_REJECTIONS[style], 2, rng)
        specs.append(
            RolloutSpec(
                category="tones",
                condition=f"tones:{style}",
                initial_prompt=prompts.NUMERIC_PUZZLES[key],
                followups=followups,
                meta={"tone": style, "puzzle": key},
            )
        )
    return specs


def _build_extended(n: int, rng: random.Random) -> list[RolloutSpec]:
    specs = []
    puzzle_keys = list(prompts.NUMERIC_PUZZLES)
    for i in range(n):
        key = puzzle_keys[i % len(puzzle_keys)]
        specs.append(
            RolloutSpec(
                category="extended",
                condition=f"extended:{key}",
                initial_prompt=prompts.NUMERIC_PUZZLES[key],
                # Fixed escalating sequence of 7 neutral rejections.
                followups=list(prompts.EXTENDED_REJECTIONS),
                meta={"puzzle": key},
            )
        )
    return specs


def _build_wildchat(
    n: int, rng: random.Random, wildchat_prompts: list[str]
) -> list[RolloutSpec]:
    specs = []
    for i in range(n):
        prompt = wildchat_prompts[i % len(wildchat_prompts)]
        followups = _take(prompts.NEUTRAL_REJECTIONS, 4, rng)
        specs.append(
            RolloutSpec(
                category="wildchat",
                condition="wildchat",
                initial_prompt=prompt,
                followups=followups,
                meta={"prompt_index": i % len(wildchat_prompts)},
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_all(
    counts: dict[str, int],
    seed: int,
    wildchat_prompts: list[str],
) -> list[RolloutSpec]:
    """Build every rollout spec for a run.

    ``counts``           : per-category number of rollouts (see config.rollout_counts).
    ``seed``             : RNG seed for reproducible variant/rejection sampling.
    ``wildchat_prompts`` : the sampled WildChat user prompts (see distress.wildchat).
    """
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []
    specs += _build_impossible_numeric(counts["impossible_numeric"], rng)
    specs += _build_triggers(counts["triggers"], rng)
    specs += _build_tones(counts["tones"], rng)
    specs += _build_extended(counts["extended"], rng)
    specs += _build_wildchat(counts["wildchat"], rng, wildchat_prompts)
    # Stable rollout ids so runs are resumable and judge results line up.
    for idx, spec in enumerate(specs):
        spec.meta["rollout_id"] = idx
    return specs
