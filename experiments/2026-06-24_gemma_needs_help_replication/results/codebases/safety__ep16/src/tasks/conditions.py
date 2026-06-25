"""The 8 evaluation conditions across 5 categories (Table 1).

A *condition* is a recipe for building rollout specs: an initial user prompt, the
number of turns, and the tone of the rejection follow-ups. The paper states "8
evaluation conditions across 5 categories" but only tabulates the 5 categories.
We resolve the 8 conditions as a documented design choice (see DESIGN.md):

  Category            | Conditions
  --------------------|------------------------------------------------------
  Impossible numeric  | impossible_numeric (3-turn, neutral)                 [1]
  Triggers            | triggers_factual (3-turn), triggers_opinion (3-turn) [2]
  Tones               | tones_aggressive, tones_disappointed, tones_sarcastic[3]
  Extended            | extended (8-turn, neutral)                           [1]
  WildChat            | wildchat (5-turn, neutral)                           [1]
  --------------------|------------------------------------------------------
  Total                                                                       8

A ``RolloutSpec`` is the fully-resolved plan for one conversation: the opening
user message plus the ordered list of rejection messages to send after each
assistant turn. ``n_turns`` == number of assistant responses == 1 + len(rejections).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from config import DATA_DIR
from src.tasks import puzzles, triggers, wildchat


def _stable_seed(seed: int, key: str) -> int:
    """Deterministic per-key seed (avoids non-reproducible builtin str hashing)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass(frozen=True)
class RolloutSpec:
    condition: str          # condition id (one of the 8)
    category: str           # one of the 5 categories
    task_id: str            # source task id
    opening: str            # first user message
    rejections: list[str]   # one message per follow-up turn
    tone: str = "neutral"
    extra: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


def _load_rejections() -> dict:
    return json.loads((DATA_DIR / "rejections.json").read_text())


# --------------------------------------------------------------------------- #
# Condition definitions
# --------------------------------------------------------------------------- #
# (condition_id, category, n_turns, tone, task_source)
CONDITIONS = [
    ("impossible_numeric", "impossible_numeric", 3, "neutral", "puzzle"),
    ("triggers_factual", "triggers", 3, "neutral", "trigger_factual"),
    ("triggers_opinion", "triggers", 3, "neutral", "trigger_opinion"),
    ("tones_aggressive", "tones", 3, "aggressive", "puzzle"),
    ("tones_disappointed", "tones", 3, "disappointed", "puzzle"),
    ("tones_sarcastic", "tones", 3, "sarcastic", "puzzle"),
    ("extended", "extended", 8, "extended_neutral", "puzzle"),
    ("wildchat", "wildchat", 5, "neutral", "wildchat"),
]

CONDITION_IDS = [c[0] for c in CONDITIONS]
CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _sample_rejections(rng: random.Random, tone: str, k: int, rej_data: dict) -> list[str]:
    pool = rej_data[tone]
    if tone == "extended_neutral":
        # Use the ordered escalating-but-neutral list, cycling if needed.
        return [pool[i % len(pool)] for i in range(k)]
    # Otherwise sample with replacement so tone varies turn to turn.
    return [rng.choice(pool) for _ in range(k)]


def build_specs_for_condition(condition_id: str, n: int, *, seed: int) -> list[RolloutSpec]:
    """Build ``n`` rollout specs for a single condition."""
    cond = next(c for c in CONDITIONS if c[0] == condition_id)
    _, category, n_turns, tone, source = cond
    rng = random.Random(_stable_seed(seed, condition_id))
    rej_data = _load_rejections()
    n_rejections = n_turns - 1

    specs: list[RolloutSpec] = []

    if source == "puzzle":
        tasks = puzzles.sample_puzzles(n, seed=seed)
        for t in tasks:
            specs.append(RolloutSpec(
                condition=condition_id, category=category, task_id=t.id, opening=t.prompt,
                rejections=_sample_rejections(rng, tone, n_rejections, rej_data), tone=tone,
                extra={"puzzle_type": t.type},
            ))
    elif source in ("trigger_factual", "trigger_opinion"):
        kind = "factual" if source.endswith("factual") else "opinion"
        all_trig = [t for t in triggers.load_triggers() if t.kind == kind]
        for i in range(n):
            t = all_trig[i % len(all_trig)] if i < len(all_trig) else rng.choice(all_trig)
            specs.append(RolloutSpec(
                condition=condition_id, category=category, task_id=t.id, opening=t.prompt,
                rejections=_sample_rejections(rng, tone, n_rejections, rej_data), tone=tone,
                extra={"trigger_kind": kind},
            ))
    elif source == "wildchat":
        wc = wildchat.load_wildchat(n, seed=seed)
        for t in wc:
            specs.append(RolloutSpec(
                condition=condition_id, category=category, task_id=t.id, opening=t.prompt,
                rejections=_sample_rejections(rng, tone, n_rejections, rej_data), tone=tone,
            ))
    else:
        raise ValueError(f"unknown task source {source!r}")

    return specs


def allocate_responses(total: int) -> dict[str, int]:
    """Split the per-model response budget across the 8 conditions.

    Design choice (DESIGN.md): the paper says 4000 responses/model "across
    evaluation categories" without giving the split. We allocate the budget
    *evenly across the 8 conditions*, then round so the totals sum to ``total``.
    """
    n_cond = len(CONDITION_IDS)
    base = total // n_cond
    alloc = {cid: base for cid in CONDITION_IDS}
    # distribute remainder
    for i in range(total - base * n_cond):
        alloc[CONDITION_IDS[i]] += 1
    return alloc


def build_all_specs(total: int, *, seed: int) -> list[RolloutSpec]:
    """Build the full set of rollout specs for one model's evaluation.

    Note ``total`` counts *conversations*, each of which yields ``n_turns``
    scored responses. The paper's "4000 responses" counts individual scored
    assistant turns; ``runner`` exposes both interpretations (see DESIGN.md).
    """
    alloc = allocate_responses(total)
    specs: list[RolloutSpec] = []
    for cid, k in alloc.items():
        specs.extend(build_specs_for_condition(cid, k, seed=seed))
    return specs
