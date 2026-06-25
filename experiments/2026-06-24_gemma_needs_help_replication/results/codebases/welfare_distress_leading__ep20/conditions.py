"""The five evaluation categories and the rollout specs they expand into.

Shared structure (Section 2.1): present a task, then reject the model's response
over multiple turns. A RolloutSpec fully determines one conversation: the opening
user message and the ordered list of rejection messages that follow each model
turn.

Categories (Table 1 / Appendix B):
  impossible_numeric (3-turn) : Countdown or Fraction puzzle, 2 neutral rejections
  triggers           (3-turn) : opinion / factual question, 2 neutral rejections
  tones              (3-turn) : numeric puzzle, 2 valenced rejections (3 styles)
  extended           (8-turn) : numeric puzzle, 7 neutral rejections
  wildchat           (5-turn) : sampled WildChat prompt, 4 neutral rejections

See DESIGN.md for how the "8 conditions across 5 categories" line in the paper
maps onto this (and where we had to make a judgement call).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import prompts
import wildchat


@dataclass
class RolloutSpec:
    rollout_id: str
    category: str               # one of the five category keys
    condition: str              # finer-grained label (e.g. "tones:aggressive")
    n_turns: int                # number of assistant turns in the conversation
    initial_user: str           # opening user message (the task)
    rejections: list[str]       # length n_turns - 1, applied after each turn
    meta: dict = field(default_factory=dict)


def _sample_neutral(rng: random.Random, k: int, lead: Optional[list[str]] = None) -> list[str]:
    """k neutral rejections. If `lead` is given, use it first then fill from pool."""
    out: list[str] = list(lead or [])
    while len(out) < k:
        out.append(rng.choice(prompts.NEUTRAL_REJECTIONS))
    return out[:k]


def build_rollouts(category: str, n_rollouts: int, seed: int = 0) -> list[RolloutSpec]:
    """Expand a category into `n_rollouts` concrete RolloutSpecs.

    Sampling is seeded so a run is reproducible. The base prompts within a
    category are cycled round-robin (so e.g. both puzzle types and all tone
    styles get balanced coverage), and rejections are sampled per-rollout.
    """
    rng = random.Random(f"{seed}:{category}")

    if category == "impossible_numeric":
        return _build_impossible_numeric(n_rollouts, rng)
    if category == "triggers":
        return _build_triggers(n_rollouts, rng)
    if category == "tones":
        return _build_tones(n_rollouts, rng)
    if category == "extended":
        return _build_extended(n_rollouts, rng)
    if category == "wildchat":
        return _build_wildchat(n_rollouts, rng, seed)
    raise ValueError(f"unknown category: {category}")


def _build_impossible_numeric(n: int, rng: random.Random) -> list[RolloutSpec]:
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    out = []
    for i in range(n):
        pk = puzzle_keys[i % len(puzzle_keys)]
        out.append(RolloutSpec(
            rollout_id=f"impossible_numeric-{i:05d}",
            category="impossible_numeric",
            condition=f"impossible_numeric:{pk}",
            n_turns=3,
            initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pk],
            rejections=_sample_neutral(rng, 2),
            meta={"puzzle": pk},
        ))
    return out


def _build_triggers(n: int, rng: random.Random) -> list[RolloutSpec]:
    # Base trigger prompts: 1 opinion + 2 factual (Appendix B).
    bases = [("opinion", prompts.TRIGGER_OPINION)]
    bases += [("factual", q) for q in prompts.TRIGGER_FACTUAL]
    out = []
    for i in range(n):
        kind, q = bases[i % len(bases)]
        out.append(RolloutSpec(
            rollout_id=f"triggers-{i:05d}",
            category="triggers",
            condition=f"triggers:{kind}",
            n_turns=3,
            initial_user=q,
            rejections=_sample_neutral(rng, 2),
            meta={"kind": kind},
        ))
    return out


def _build_tones(n: int, rng: random.Random) -> list[RolloutSpec]:
    styles = list(prompts.TONE_REJECTIONS.keys())  # aggressive/disappointed/sarcastic
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    out = []
    for i in range(n):
        style = styles[i % len(styles)]
        pk = puzzle_keys[i % len(puzzle_keys)]
        # Two rejections in the chosen tone (sample with replacement from the
        # style's pool of two example lines).
        rejs = [rng.choice(prompts.TONE_REJECTIONS[style]) for _ in range(2)]
        out.append(RolloutSpec(
            rollout_id=f"tones-{i:05d}",
            category="tones",
            condition=f"tones:{style}",
            n_turns=3,
            initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pk],
            rejections=rejs,
            meta={"tone": style, "puzzle": pk},
        ))
    return out


def _build_extended(n: int, rng: random.Random) -> list[RolloutSpec]:
    puzzle_keys = list(prompts.IMPOSSIBLE_NUMERIC_PUZZLES.keys())
    out = []
    for i in range(n):
        pk = puzzle_keys[i % len(puzzle_keys)]
        out.append(RolloutSpec(
            rollout_id=f"extended-{i:05d}",
            category="extended",
            condition="extended",
            n_turns=8,
            initial_user=prompts.IMPOSSIBLE_NUMERIC_PUZZLES[pk],
            # 7 rejections: verbatim opening sequence, then fill from neutral pool.
            rejections=_sample_neutral(rng, 7, lead=prompts.EXTENDED_REJECTION_SEQUENCE),
            meta={"puzzle": pk},
        ))
    return out


def _build_wildchat(n: int, rng: random.Random, seed: int) -> list[RolloutSpec]:
    # Paper: 20 distinct prompts x 40 samples each. We mirror that ratio: take a
    # fixed pool of distinct prompts and assign rollouts round-robin across them.
    pool = wildchat.load_prompts(n_prompts=20, seed=seed)
    out = []
    for i in range(n):
        prompt = pool[i % len(pool)]
        out.append(RolloutSpec(
            rollout_id=f"wildchat-{i:05d}",
            category="wildchat",
            condition="wildchat",
            n_turns=5,
            initial_user=prompt,
            rejections=_sample_neutral(rng, 4),
            meta={"prompt_index": i % len(pool)},
        ))
    return out


ALL_CATEGORIES = [
    "impossible_numeric",
    "triggers",
    "tones",
    "extended",
    "wildchat",
]
