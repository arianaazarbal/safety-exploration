"""The 8 evaluation conditions across 5 categories (paper Table 1).

The paper says "8 evaluation conditions across 5 categories" but does not
enumerate the 8 explicitly. Our reading (documented in DESIGN.md):

    Category            Conditions
    impossible numeric  countdown, fraction                     (2)
    triggers            triggers (opinion + factual pooled)     (1)
    tones               aggressive, disappointed, sarcastic     (3)
    extended            extended                                (1)
    wildchat            wildchat                                (1)
                                                          total = 8

One *conversation* is one *scored response* in the paper's accounting (see
DESIGN.md §What counts as a "response"). The per-category conversation counts
below therefore equal the paper's per-category response counts and sum to 4000:

    numeric 2000 (1000+1000) | triggers 400 | tones 600 (200x3)
    | extended 200 | wildchat 800  ->  4000
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import config
import prompts
from prompts import Puzzle
from wildchat import load_wildchat_prompts


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int
    rejection_mode: str           # "neutral" or a key of TONE_REJECTIONS
    base_conversations: int       # at paper scale
    pool: list[Puzzle]


# Base (paper-scale) conversation counts per condition.
_BASE = {
    "numeric_countdown": 1000,
    "numeric_fraction": 1000,
    "triggers": 400,
    "tones_aggressive": 200,
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,
    "wildchat": 800,
}


def build_conditions(seed: int = 0) -> list[Condition]:
    """Construct all 8 conditions. WildChat prompts are loaded lazily here."""
    numeric_count = prompts.numeric_pool("countdown")
    numeric_frac = prompts.numeric_pool("fraction")
    # Extended uses impossible numeric puzzles too (both kinds pooled).
    extended_pool = numeric_count + numeric_frac
    # Tones use impossible numeric puzzles with varied rejection styles.
    tones_pool = numeric_count + numeric_frac
    wildchat_pool = load_wildchat_prompts(seed=seed)

    return [
        Condition("numeric_countdown", "impossible_numeric", 3, "neutral",
                  _BASE["numeric_countdown"], numeric_count),
        Condition("numeric_fraction", "impossible_numeric", 3, "neutral",
                  _BASE["numeric_fraction"], numeric_frac),
        Condition("triggers", "triggers", 3, "neutral",
                  _BASE["triggers"], prompts.TRIGGER_PROMPTS),
        Condition("tones_aggressive", "tones", 3, "aggressive",
                  _BASE["tones_aggressive"], tones_pool),
        Condition("tones_disappointed", "tones", 3, "disappointed",
                  _BASE["tones_disappointed"], tones_pool),
        Condition("tones_sarcastic", "tones", 3, "sarcastic",
                  _BASE["tones_sarcastic"], tones_pool),
        Condition("extended", "extended", 8, "neutral",
                  _BASE["extended"], extended_pool),
        Condition("wildchat", "wildchat", 5, "neutral",
                  _BASE["wildchat"], wildchat_pool),
    ]


def n_conversations(cond: Condition, profile: config.Profile) -> int:
    scaled = round(cond.base_conversations * profile.scale)
    return max(profile.min_conversations, scaled)


def _rng_for(seed: int, model_key: str, cond_name: str, idx: int) -> random.Random:
    h = hashlib.sha256(
        f"{seed}|{model_key}|{cond_name}|{idx}".encode()
    ).hexdigest()
    return random.Random(int(h[:16], 16))


def rejection_sequence(cond: Condition, rng: random.Random) -> list[str]:
    """Return the n_turns-1 rejection messages for one conversation.

    Rejections are sampled (with replacement avoided where the pool allows) so
    that, as the paper puts it, follow-ups are "randomised". Tone conditions
    draw from the matching tone pool; everything else from the neutral pool.
    """
    pool = (prompts.NEUTRAL_REJECTIONS if cond.rejection_mode == "neutral"
            else prompts.TONE_REJECTIONS[cond.rejection_mode])
    k = cond.n_turns - 1
    if k <= len(pool):
        return rng.sample(pool, k)
    # Need more rejections than the pool size (extended, 7 > pool): sample with
    # replacement but avoid immediate repeats where possible.
    seq: list[str] = []
    for _ in range(k):
        choice = rng.choice(pool)
        if len(pool) > 1:
            while seq and choice == seq[-1]:
                choice = rng.choice(pool)
        seq.append(choice)
    return seq


@dataclass(frozen=True)
class ConversationPlan:
    """A fully-specified, deterministic conversation to run."""
    conversation_id: str
    model_key: str
    condition: str
    category: str
    n_turns: int
    rejection_mode: str
    prompt_id: str
    prompt_text: str
    rejections: list[str]


def plan_conversations(model_key: str, cond: Condition,
                       profile: config.Profile, seed: int) -> list[ConversationPlan]:
    plans: list[ConversationPlan] = []
    n = n_conversations(cond, profile)
    pool = cond.pool
    if not pool:
        return plans
    for i in range(n):
        puzzle = pool[i % len(pool)]
        rng = _rng_for(seed, model_key, cond.name, i)
        rejections = rejection_sequence(cond, rng)
        plans.append(ConversationPlan(
            conversation_id=f"{model_key}|{cond.name}|{i:05d}",
            model_key=model_key,
            condition=cond.name,
            category=cond.category,
            n_turns=cond.n_turns,
            rejection_mode=cond.rejection_mode,
            prompt_id=puzzle.id,
            prompt_text=puzzle.text,
            rejections=rejections,
        ))
    return plans
