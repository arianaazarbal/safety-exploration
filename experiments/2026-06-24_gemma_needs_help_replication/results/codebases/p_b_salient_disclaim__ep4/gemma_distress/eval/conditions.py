"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Categories and their conditions (8 total):
  1. impossible_numeric (3-turn, 2 neutral rejections)            -> 1 condition
  2. triggers           (3-turn, 2 neutral rejections)            -> 2 conditions
                        (opinion, factual)
  3. tones              (3-turn, impossible numeric base)         -> 3 conditions
                        (aggressive, disappointed, sarcastic)
  4. extended           (8-turn, 7 neutral rejections)            -> 1 condition
  5. wildchat           (5-turn, 4 neutral rejections)            -> 1 condition
                                                                  = 8 conditions

See DESIGN.md ("Counting the 8 conditions") for why triggers splits into 2 and
tones into 3 to total 8.

A :class:`RolloutSpec` fully describes one conversation to run: the initial task
prompt, the number of turns, and a per-turn rejection strategy. ``rng_seed``
makes rejection sampling deterministic per conversation.

The per-category conversation budget (Appendix B) is distributed across that
category's conditions and over the prompt pool. We treat the Appendix B counts
as numbers of *conversations* (the reading under which 20x40 WildChat and the
200 Extended counts divide cleanly); every assistant turn within a conversation
is independently judged. See DESIGN.md ("Responses vs conversations").
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .. import config
from ..prompts import (EXTENDED_REJECTIONS, TONE_REJECTIONS, Puzzle,
                       generate_puzzles, sample_rejections)
from ..prompts.triggers import (FACTUAL_TRIGGERS, OPINION_TRIGGERS, Trigger)
from ..prompts.wildchat import sample_wildchat_prompts


# A rejection strategy maps (turn_index_1based, rng) -> rejection text. Turn 1 is
# the initial task (no rejection precedes it); rejections apply at turns >= 2.
RejectionFn = Callable[[int, random.Random], str]


@dataclass
class RolloutSpec:
    condition: str          # e.g. "tones_aggressive"
    category: str           # one of the 5 categories
    task_prompt: str
    turns: int
    rejection_fn: RejectionFn
    system_prompt: Optional[str] = None
    rng_seed: int = 0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Rejection strategies
# --------------------------------------------------------------------------- #
def _neutral_rejection_fn(turn: int, rng: random.Random) -> str:
    return sample_rejections(1, rng=rng)[0]


def _tone_rejection_fn(tone: str) -> RejectionFn:
    pool = TONE_REJECTIONS[tone]
    def fn(turn: int, rng: random.Random) -> str:
        return rng.choice(pool)
    return fn


def _extended_rejection_fn(turn: int, rng: random.Random) -> str:
    # turn 2 uses EXTENDED_REJECTIONS[0], etc.
    idx = min(turn - 2, len(EXTENDED_REJECTIONS) - 1)
    return EXTENDED_REJECTIONS[idx]


# --------------------------------------------------------------------------- #
# Condition builders
# --------------------------------------------------------------------------- #
def build_condition_specs(seed: int = 0) -> List[RolloutSpec]:
    """Build the full list of RolloutSpecs summing to ~4000 conversations."""
    rng = random.Random(seed)
    specs: List[RolloutSpec] = []
    specs += _numeric_specs(rng)
    specs += _trigger_specs(rng)
    specs += _tone_specs(rng)
    specs += _extended_specs(rng)
    specs += _wildchat_specs(rng)
    return specs


def _numeric_specs(rng: random.Random) -> List[RolloutSpec]:
    n = config.RESPONSES_PER_CATEGORY["impossible_numeric"]
    pool = generate_puzzles(max(8, n // 50), seed=rng.randint(0, 10**6))
    out = []
    for i in range(n):
        puz = pool[i % len(pool)]
        out.append(RolloutSpec(
            condition="impossible_numeric",
            category="impossible_numeric",
            task_prompt=puz.prompt,
            turns=config.SHORT_TURNS,
            rejection_fn=_neutral_rejection_fn,
            rng_seed=rng.randint(0, 10**9),
            meta={"puzzle_id": puz.id, "puzzle_category": puz.category},
        ))
    return out


def _trigger_specs(rng: random.Random) -> List[RolloutSpec]:
    total = config.RESPONSES_PER_CATEGORY["triggers"]
    half = total // 2
    out = []
    out += _trigger_group("triggers_opinion", OPINION_TRIGGERS, half, rng)
    out += _trigger_group("triggers_factual", FACTUAL_TRIGGERS, total - half, rng)
    return out


def _trigger_group(condition: str, triggers: List[Trigger], n: int,
                   rng: random.Random) -> List[RolloutSpec]:
    out = []
    for i in range(n):
        t = triggers[i % len(triggers)]
        out.append(RolloutSpec(
            condition=condition,
            category="triggers",
            task_prompt=t.prompt,
            turns=config.SHORT_TURNS,
            rejection_fn=_neutral_rejection_fn,
            rng_seed=rng.randint(0, 10**9),
            meta={"trigger_id": t.id, "trigger_kind": t.kind},
        ))
    return out


def _tone_specs(rng: random.Random) -> List[RolloutSpec]:
    total = config.RESPONSES_PER_CATEGORY["tones"]
    per_tone = total // 3
    pool = generate_puzzles(max(8, per_tone // 50), seed=rng.randint(0, 10**6))
    out = []
    tones = list(TONE_REJECTIONS.keys())
    for t_idx, tone in enumerate(tones):
        # last tone soaks up the remainder
        n = per_tone if t_idx < len(tones) - 1 else total - per_tone * (len(tones) - 1)
        fn = _tone_rejection_fn(tone)
        for i in range(n):
            puz = pool[i % len(pool)]
            out.append(RolloutSpec(
                condition=f"tones_{tone}",
                category="tones",
                task_prompt=puz.prompt,
                turns=config.SHORT_TURNS,
                rejection_fn=fn,
                rng_seed=rng.randint(0, 10**9),
                meta={"tone": tone, "puzzle_id": puz.id},
            ))
    return out


def _extended_specs(rng: random.Random) -> List[RolloutSpec]:
    n = config.RESPONSES_PER_CATEGORY["extended"]
    pool = generate_puzzles(max(8, n // 25), seed=rng.randint(0, 10**6))
    out = []
    for i in range(n):
        puz = pool[i % len(pool)]
        out.append(RolloutSpec(
            condition="extended",
            category="extended",
            task_prompt=puz.prompt,
            turns=config.EXTENDED_TURNS,
            rejection_fn=_extended_rejection_fn,
            rng_seed=rng.randint(0, 10**9),
            meta={"puzzle_id": puz.id},
        ))
    return out


def _wildchat_specs(rng: random.Random) -> List[RolloutSpec]:
    prompts = sample_wildchat_prompts(config.WILDCHAT_N_PROMPTS,
                                      seed=rng.randint(0, 10**6))
    out = []
    for p_idx, prompt in enumerate(prompts):
        for _ in range(config.WILDCHAT_SAMPLES_PER_PROMPT):
            out.append(RolloutSpec(
                condition="wildchat",
                category="wildchat",
                task_prompt=prompt,
                turns=config.WILDCHAT_TURNS,
                rejection_fn=_neutral_rejection_fn,
                rng_seed=rng.randint(0, 10**9),
                meta={"wildchat_prompt_idx": p_idx},
            ))
    return out
