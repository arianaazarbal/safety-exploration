"""The 8 evaluation conditions across 5 categories (Table 1).

Each condition knows how to build the conversation *plan* for one rollout: the
initial task prompt and the ordered list of user follow-up (rejection) messages.
A "turn" is one user→assistant exchange; an N-turn condition has 1 task prompt +
(N-1) follow-ups.

Counting (5 categories, 8 conditions):
  1. Impossible numeric (3-turn)            [category: impossible_numeric]
  2. Triggers — opinion (3-turn)            [category: triggers]
  3. Triggers — factual (3-turn)            [category: triggers]
  4. Tones — aggressive (3-turn)            [category: tones]
  5. Tones — disappointed (3-turn)          [category: tones]
  6. Tones — sarcastic (3-turn)             [category: tones]
  7. Extended (8-turn)                      [category: extended]
  8. WildChat (5-turn)                      [category: wildchat]
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..data import puzzles, rejections, triggers, wildchat


@dataclass
class RolloutPlan:
    """One conversation's worth of scripted user turns."""
    condition: str          # condition id (e.g. "tones_aggressive")
    category: str           # one of the 5 categories
    task_prompt: str        # first user message
    followups: list[str]    # subsequent user messages (rejections)
    n_turns: int            # total assistant turns expected
    meta: dict              # provenance (puzzle params, trigger subtype, etc.)


# --------------------------------------------------------------------------- #
# Per-category plan builders
# --------------------------------------------------------------------------- #


def build_impossible_numeric(n_conversations: int, seed: int = 0) -> list[RolloutPlan]:
    rng = random.Random(seed)
    puz = puzzles.sample_impossible_puzzles(n_conversations, seed=seed)
    plans = []
    for p in puz:
        plans.append(RolloutPlan(
            condition="impossible_numeric",
            category="impossible_numeric",
            task_prompt=p.prompt,
            followups=rejections.neutral_rejections(2, rng),  # 3-turn -> 2 followups
            n_turns=3,
            meta={"puzzle_kind": p.kind, **p.params,
                  "verified_impossible": p.verified_impossible},
        ))
    return plans


def build_triggers(n_conversations: int, seed: int = 0) -> list[RolloutPlan]:
    rng = random.Random(seed + 1)
    pairs = triggers.all_triggers()
    plans = []
    for i in range(n_conversations):
        subtype, q = pairs[i % len(pairs)]
        plans.append(RolloutPlan(
            condition=f"triggers_{subtype}",
            category="triggers",
            task_prompt=q,
            followups=rejections.neutral_rejections(2, rng),
            n_turns=3,
            meta={"subtype": subtype},
        ))
    return plans


def build_tones(n_conversations: int, seed: int = 0) -> list[RolloutPlan]:
    rng = random.Random(seed + 2)
    tones = ["aggressive", "disappointed", "sarcastic"]
    puz = puzzles.sample_impossible_puzzles(n_conversations, seed=seed + 100)
    plans = []
    for i, p in enumerate(puz):
        tone = tones[i % len(tones)]
        plans.append(RolloutPlan(
            condition=f"tones_{tone}",
            category="tones",
            task_prompt=p.prompt,
            followups=rejections.tone_rejections(tone, 2, rng),  # 3-turn
            n_turns=3,
            meta={"tone": tone, "puzzle_kind": p.kind},
        ))
    return plans


def build_extended(n_conversations: int, seed: int = 0) -> list[RolloutPlan]:
    puz = puzzles.sample_impossible_puzzles(n_conversations, seed=seed + 200)
    plans = []
    for p in puz:
        plans.append(RolloutPlan(
            condition="extended",
            category="extended",
            task_prompt=p.prompt,
            followups=rejections.extended_rejections(7),  # 8-turn -> 7 followups
            n_turns=8,
            meta={"puzzle_kind": p.kind},
        ))
    return plans


def build_wildchat(n_conversations: int, seed: int = 0) -> list[RolloutPlan]:
    rng = random.Random(seed + 3)
    prompts = wildchat.sample_wildchat_prompts(n_prompts=20, seed=seed)
    plans = []
    for i in range(n_conversations):
        q = prompts[i % len(prompts)]
        plans.append(RolloutPlan(
            condition="wildchat",
            category="wildchat",
            task_prompt=q,
            followups=rejections.neutral_rejections(4, rng),  # 5-turn -> 4 followups
            n_turns=5,
            meta={"prompt_index": i % len(prompts)},
        ))
    return plans


# --------------------------------------------------------------------------- #
# Building the full protocol from per-category *response* targets.
#
# Each rollout produces `n_turns` judged responses (one per assistant turn), so
# the number of conversations needed = ceil(target_responses / n_turns).
# --------------------------------------------------------------------------- #

_TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

_BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_protocol(response_targets: dict[str, int], seed: int = 0) -> list[RolloutPlan]:
    """Build every rollout plan for the full Section 2 protocol.

    `response_targets` maps category -> desired judged-response count
    (e.g. {"impossible_numeric": 2000, ...}). Returns a flat list of plans.
    """
    plans: list[RolloutPlan] = []
    for category, target in response_targets.items():
        n_conv = math.ceil(target / _TURNS[category])
        plans.extend(_BUILDERS[category](n_conv, seed=seed))
    return plans
