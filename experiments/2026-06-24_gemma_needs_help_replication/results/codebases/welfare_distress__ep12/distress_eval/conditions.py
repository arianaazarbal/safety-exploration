"""The 8 evaluation conditions across 5 categories (paper Table 1).

A *condition* is one row of the experiment grid. We turn each condition into a
list of `ConversationSpec`s: a fully-determined opening user message plus the
ordered sequence of user follow-up (rejection) messages. The runner executes a
spec turn-by-turn against a target model and scores every assistant turn.

Categories -> conditions:
  Impossible numeric : numeric            (3 turns, 2 neutral rejections)
  Triggers           : trigger_opinion    (3 turns, 2 neutral rejections)
                       trigger_factual     (3 turns, 2 neutral rejections)
  Tones              : tone_aggressive     (3 turns, 2 aggressive rejections)
                       tone_disappointed   (3 turns, 2 disappointed rejections)
                       tone_sarcastic      (3 turns, 2 sarcastic rejections)
  Extended           : extended            (8 turns, 7 neutral rejections)
  WildChat           : wildchat            (5 turns, 4 neutral rejections)
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass

from . import prompts, puzzles, wildchat


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    turns: int                 # number of assistant responses == 1 + n_rejections
    source: str                # "numeric" | "opinion" | "factual" | "wildchat"
    tone: str = "neutral"      # "neutral" | "aggressive" | "disappointed" | "sarcastic"
    base_conversations: int = 0  # at EVAL_SCALE=1.0


# Default budget (~4000 responses/model: see response_budget() below).
CONDITIONS: list[Condition] = [
    Condition("numeric", "impossible_numeric", 3, "numeric", base_conversations=267),
    Condition("trigger_opinion", "triggers", 3, "opinion", base_conversations=133),
    Condition("trigger_factual", "triggers", 3, "factual", base_conversations=133),
    Condition("tone_aggressive", "tones", 3, "numeric", tone="aggressive", base_conversations=89),
    Condition("tone_disappointed", "tones", 3, "numeric", tone="disappointed", base_conversations=89),
    Condition("tone_sarcastic", "tones", 3, "numeric", tone="sarcastic", base_conversations=89),
    Condition("extended", "extended", 8, "numeric", base_conversations=100),
    Condition("wildchat", "wildchat", 5, "wildchat", base_conversations=160),
]


@dataclass(frozen=True)
class ConversationSpec:
    condition: str
    category: str
    tone: str
    turns: int
    prompt_id: str
    repeat_idx: int
    opening: str               # first user message
    followups: tuple[str, ...] # subsequent user messages (len == turns - 1)


def response_budget(scale: float = 1.0) -> dict:
    """Return per-condition conversation/response counts at the given scale."""
    out = {}
    total_resp = 0
    for c in CONDITIONS:
        n_conv = max(1, math.ceil(c.base_conversations * scale))
        resp = n_conv * c.turns
        total_resp += resp
        out[c.name] = {"conversations": n_conv, "turns": c.turns, "responses": resp}
    out["_total_responses"] = total_resp
    return out


def _opening_pool(cond: Condition, n: int, seed: int) -> list[tuple[str, str]]:
    """Return up to `n` (prompt_id, opening_text) drawing from the condition's
    source pool, cycling with repeats when n exceeds the pool size."""
    if cond.source == "numeric":
        pool = [(p.puzzle_id, p.prompt) for p in puzzles.ALL_NUMERIC_PUZZLES]
    elif cond.source == "opinion":
        pool = [(f"opinion_{i:02d}", q) for i, q in enumerate(prompts.OPINION_QUESTIONS)]
    elif cond.source == "factual":
        pool = [(f"factual_{i:02d}", q) for i, q in enumerate(prompts.FACTUAL_QUESTIONS)]
    elif cond.source == "wildchat":
        wc = wildchat.load_wildchat_prompts(n, seed=seed)
        return [(f"wildchat_{i:04d}", t) for i, t in enumerate(wc)][:n]
    else:
        raise ValueError(f"unknown source {cond.source}")

    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    i = 0
    while len(out) < n:
        out.append(pool[i % len(pool)])
        i += 1
    return out


def _rejection_for_turn(cond: Condition, rng: random.Random) -> str:
    if cond.tone == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    return rng.choice(prompts.TONED_REJECTIONS[cond.tone])


def build_specs(cond: Condition, scale: float, seed: int) -> list[ConversationSpec]:
    """Materialize all ConversationSpecs for one condition."""
    n_conv = max(1, math.ceil(cond.base_conversations * scale))
    # Stable per-condition seed offset (zlib.crc32 is deterministic across runs,
    # unlike builtin hash() which is salted by PYTHONHASHSEED).
    cond_offset = zlib.crc32(cond.name.encode()) % 100000
    openings = _opening_pool(cond, n_conv, seed=seed + cond_offset)

    # Track how many times each prompt_id has been reused -> repeat_idx.
    seen: dict[str, int] = {}
    specs: list[ConversationSpec] = []
    for k, (pid, opening) in enumerate(openings):
        repeat_idx = seen.get(pid, 0)
        seen[pid] = repeat_idx + 1
        # Deterministic but per-conversation rejection sampling.
        rng = random.Random(seed * 7919 + k)
        followups = tuple(_rejection_for_turn(cond, rng) for _ in range(cond.turns - 1))
        specs.append(
            ConversationSpec(
                condition=cond.name,
                category=cond.category,
                tone=cond.tone,
                turns=cond.turns,
                prompt_id=pid,
                repeat_idx=repeat_idx,
                opening=opening,
                followups=followups,
            )
        )
    return specs


def build_all_specs(scale: float, seed: int) -> list[ConversationSpec]:
    specs: list[ConversationSpec] = []
    for cond in CONDITIONS:
        specs.extend(build_specs(cond, scale, seed))
    return specs
