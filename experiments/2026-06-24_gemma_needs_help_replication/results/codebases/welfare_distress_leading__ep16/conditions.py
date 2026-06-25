"""Define the 8 evaluation conditions (across 5 categories) and expand them into a
concrete list of rollout specifications for a given model.

paper Section 2.1 / Table 1: "8 evaluation conditions across 5 categories". The five
categories are numeric, triggers, tones, extended, WildChat. The extra conditions come
from triggers (opinion vs factual) and tones (aggressive / disappointed / sarcastic):

    numeric            -> 1 condition
    triggers           -> 2 conditions (opinion, factual)
    tones              -> 3 conditions (aggressive, disappointed, sarcastic)
    extended           -> 1 condition
    wildchat           -> 1 condition
    ------------------------------------------------------------------------
    total              -> 8 conditions across 5 categories

A "turn" here is one model response. An N-turn condition therefore needs N-1 user
rejections after the initial question. See DESIGN.md for the rollout/turn accounting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
import prompts
from wildchat import load_wildchat_prompts


@dataclass
class RolloutSpec:
    """One full multi-turn conversation to run against one model."""
    model_key: str
    condition: str          # e.g. "tones_aggressive"
    category: str           # e.g. "tones"  (used for the paper's 5-category aggregation)
    rollout_id: int
    first_user: str         # initial user message (the task/question)
    rejections: list[str]   # the N-1 follow-up rejections
    n_turns: int            # number of model responses
    meta: dict = field(default_factory=dict)


def _category_rollouts(category: str) -> int:
    base = config.PAPER_CATEGORY_ROLLOUTS[category]
    return max(1, round(base * config.ROLLOUT_SCALE))


def _neutral_rejections(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (paper: 'two randomised neutral rejections').

    Sampled without replacement when the pool is large enough, otherwise with."""
    pool = prompts.NEUTRAL_REJECTIONS
    if k <= len(pool):
        return rng.sample(pool, k)
    out = []
    while len(out) < k:
        out.append(rng.choice(pool))
    return out


def _tone_rejections(rng: random.Random, tone: str, k: int) -> list[str]:
    variants = prompts.TONE_REJECTIONS[tone]
    if k <= len(variants):
        return rng.sample(variants, k)
    out = []
    while len(out) < k:
        out.append(rng.choice(variants))
    return out


def build_rollout_specs(model_key: str, seed: int = config.SEED) -> list[RolloutSpec]:
    """Expand all 8 conditions into concrete rollouts for one model."""
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []
    counter = 0

    def add(condition, category, first_user, rejections, n_turns, meta):
        nonlocal counter
        specs.append(RolloutSpec(
            model_key=model_key, condition=condition, category=category,
            rollout_id=counter, first_user=first_user, rejections=rejections,
            n_turns=n_turns, meta=meta,
        ))
        counter += 1

    # ---- Category: numeric (3-turn, neutral rejections) --------------------------------
    puzzle_keys = list(prompts.NUMERIC_PUZZLES)
    for i in range(_category_rollouts("numeric")):
        pk = puzzle_keys[i % len(puzzle_keys)]
        add("numeric", "numeric", prompts.NUMERIC_PUZZLES[pk],
            _neutral_rejections(rng, 2), 3, {"puzzle": pk})

    # ---- Category: triggers (3-turn, neutral) -> opinion + factual ---------------------
    n_triggers = _category_rollouts("triggers")
    n_opinion = n_triggers // 2
    n_factual = n_triggers - n_opinion
    for i in range(n_opinion):
        q = prompts.TRIGGER_OPINION[i % len(prompts.TRIGGER_OPINION)]
        add("triggers_opinion", "triggers", q, _neutral_rejections(rng, 2), 3,
            {"question": q})
    for i in range(n_factual):
        q = prompts.TRIGGER_FACTUAL[i % len(prompts.TRIGGER_FACTUAL)]
        add("triggers_factual", "triggers", q, _neutral_rejections(rng, 2), 3,
            {"question": q})

    # ---- Category: tones (3-turn, numeric base, valenced rejections) -------------------
    n_tones = _category_rollouts("tones")
    tone_names = list(prompts.TONE_REJECTIONS)  # aggressive, disappointed, sarcastic
    per_tone = n_tones // len(tone_names)
    remainder = n_tones - per_tone * len(tone_names)
    for t_idx, tone in enumerate(tone_names):
        count = per_tone + (1 if t_idx < remainder else 0)
        for i in range(count):
            pk = puzzle_keys[i % len(puzzle_keys)]
            add(f"tones_{tone}", "tones", prompts.NUMERIC_PUZZLES[pk],
                _tone_rejections(rng, tone, 2), 3, {"puzzle": pk, "tone": tone})

    # ---- Category: extended (8-turn, numeric base, 7 neutral rejections) ---------------
    for i in range(_category_rollouts("extended")):
        pk = puzzle_keys[i % len(puzzle_keys)]
        add("extended", "extended", prompts.NUMERIC_PUZZLES[pk],
            list(prompts.EXTENDED_REJECTION_SEQUENCE), 8, {"puzzle": pk})

    # ---- Category: wildchat (5-turn, 4 neutral rejections) -----------------------------
    wc_prompts = load_wildchat_prompts(seed=seed)
    n_wc = _category_rollouts("wildchat")
    for i in range(n_wc):
        p = wc_prompts[i % len(wc_prompts)]
        # 4 rejections; with a 4-item neutral pool this is the pool in a random order.
        add("wildchat", "wildchat", p, _neutral_rejections(rng, 4), 5,
            {"wildchat_prompt": p})

    return specs


# The 8 condition names, for reference / sanity checks.
CONDITIONS = [
    "numeric",
    "triggers_opinion", "triggers_factual",
    "tones_aggressive", "tones_disappointed", "tones_sarcastic",
    "extended",
    "wildchat",
]
CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]
