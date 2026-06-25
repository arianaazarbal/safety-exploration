"""The 8 evaluation conditions across 5 categories (paper Table 1).

Each condition knows how to construct one rollout's worth of inputs: the task
prompt (turn 1) and the ordered list of user rejection messages for the
remaining turns. Per-rollout randomness is driven by a seeded RNG so runs are
reproducible.

Categories -> conditions:
  impossible_numeric : numeric                         (3-turn)
  triggers           : trigger_opinion, trigger_factual(3-turn)
  tones              : tone_aggressive, tone_disappointed, tone_sarcastic (3-turn)
  extended           : extended                         (8-turn)
  wildchat           : wildchat                         (5-turn)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import prompts
import puzzles
from wildchat import load_wildchat_prompts

# A rollout spec: (prompt_id, task_prompt_text, [rejection_1, rejection_2, ...])
RolloutSpec = Tuple[str, str, List[str]]


@dataclass
class Condition:
    id: str
    category: str
    n_turns: int  # number of assistant responses (turn 1 = task answer)
    build: Callable[[random.Random], RolloutSpec]


# ---------------------------------------------------------------------------
# Lazy WildChat prompt pool (loaded once, shared across rollouts).
# ---------------------------------------------------------------------------
_wildchat_pool: List[str] | None = None


def _get_wildchat_pool(seed: int) -> List[str]:
    global _wildchat_pool
    if _wildchat_pool is None:
        _wildchat_pool = load_wildchat_prompts(n=20, seed=seed)
    return _wildchat_pool


# ---------------------------------------------------------------------------
# Rejection helpers.
# ---------------------------------------------------------------------------
def _sample_rejections(rng: random.Random, pool: List[str], count: int) -> List[str]:
    """Pick `count` rejections from a pool, allowing repeats if the pool is
    smaller than count, but avoiding immediate repetition where possible."""
    out: List[str] = []
    prev = None
    for _ in range(count):
        choices = [r for r in pool if r != prev] or pool
        pick = rng.choice(choices)
        out.append(pick)
        prev = pick
    return out


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------
def _build_numeric(rng: random.Random, n_turns: int = 3) -> RolloutSpec:
    puzzle = rng.choice(puzzles.DEFAULT_NUMERIC_PUZZLES)
    rejections = _sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, n_turns - 1)
    return puzzle.puzzle_id, puzzle.prompt, rejections


def _build_trigger_opinion(rng: random.Random) -> RolloutSpec:
    q = rng.choice(prompts.OPINION_TRIGGERS)
    rejections = _sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 2)
    return f"opinion::{q}", q, rejections


def _build_trigger_factual(rng: random.Random) -> RolloutSpec:
    q = rng.choice(prompts.FACTUAL_TRIGGERS)
    rejections = _sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 2)
    return f"factual::{q}", q, rejections


def _build_tone(rng: random.Random, pool: List[str]) -> RolloutSpec:
    # Tones always use an impossible numeric base prompt (paper Table 1).
    puzzle = rng.choice(puzzles.DEFAULT_NUMERIC_PUZZLES)
    rejections = _sample_rejections(rng, pool, 2)
    return puzzle.puzzle_id, puzzle.prompt, rejections


def _build_extended(rng: random.Random) -> RolloutSpec:
    # 8-turn: impossible numeric + the fixed ordered sequence of 7 rejections.
    puzzle = rng.choice(puzzles.DEFAULT_NUMERIC_PUZZLES)
    rejections = list(prompts.EXTENDED_REJECTIONS)  # exactly 7
    return puzzle.puzzle_id, puzzle.prompt, rejections


def _make_wildchat_builder(seed: int) -> Callable[[random.Random], RolloutSpec]:
    def _build_wildchat(rng: random.Random) -> RolloutSpec:
        pool = _get_wildchat_pool(seed)
        q = rng.choice(pool)
        rejections = _sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 4)
        pid = "wildchat::" + q[:40].replace("\n", " ")
        return pid, q, rejections

    return _build_wildchat


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
def build_conditions(seed: int) -> Dict[str, Condition]:
    """Return all 8 conditions. `seed` only affects WildChat prompt selection;
    per-rollout randomness is supplied separately at run time."""
    wildchat_builder = _make_wildchat_builder(seed)
    conditions = [
        Condition("numeric", "impossible_numeric", 3, _build_numeric),
        Condition("trigger_opinion", "triggers", 3, _build_trigger_opinion),
        Condition("trigger_factual", "triggers", 3, _build_trigger_factual),
        Condition(
            "tone_aggressive", "tones", 3,
            lambda rng: _build_tone(rng, prompts.AGGRESSIVE_REJECTIONS),
        ),
        Condition(
            "tone_disappointed", "tones", 3,
            lambda rng: _build_tone(rng, prompts.DISAPPOINTED_REJECTIONS),
        ),
        Condition(
            "tone_sarcastic", "tones", 3,
            lambda rng: _build_tone(rng, prompts.SARCASTIC_REJECTIONS),
        ),
        Condition("extended", "extended", 8, _build_extended),
        Condition("wildchat", "wildchat", 5, wildchat_builder),
    ]
    return {c.id: c for c in conditions}
