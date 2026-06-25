"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

A `Condition` is a declarative spec the rollout protocol expands into concrete
multi-turn conversations:

  Category            Conditions                      Rollouts (paper budget)
  ------------------  ------------------------------  -----------------------
  impossible_numeric  1 (3-turn, neutral)             2000
  triggers            2 (opinion, factual; 3-turn)    400  (200 each)
  tones               3 (aggressive/disappointed/     600  (200 each)
                         sarcastic; 3-turn numeric)
  extended            1 (8-turn, neutral numeric)     200
  wildchat            1 (5-turn, neutral)             800  (20 prompts x 40)
  ------------------  ------------------------------  -----------------------
  total                                               4000

The "8 conditions across 5 categories" count comes from splitting triggers into
opinion+factual (2) and tones into its three valences (3). See DESIGN.md.

Each condition carries a `rejection_fn(turn_index, rng) -> str` that produces the
user follow-up for a given 0-based follow-up index, and a `seeds` list of task
prompts cycled across rollouts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import SECTION2_BUDGET, TURNS, WILDCHAT_SAMPLES_PER_PROMPT
from ..prompts import rejections as rej
from ..prompts.puzzles import Puzzle, impossible_numeric_puzzles
from ..prompts.triggers import trigger_questions
from ..prompts.wildchat import sample_wildchat_prompts


@dataclass
class TaskSeed:
    task_id: str
    prompt: str


@dataclass
class Condition:
    category: str
    key: str                       # unique condition key, e.g. "tones:sarcastic".
    n_turns: int                   # total assistant turns (incl. first answer).
    n_rollouts: int                # number of conversations to sample.
    seeds: list[TaskSeed]
    rejection_fn: Callable[[int, random.Random], str]
    system_prompt: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def seed_for(self, rollout_index: int) -> TaskSeed:
        return self.seeds[rollout_index % len(self.seeds)]


def _seeds_from_puzzles(puzzles: list[Puzzle]) -> list[TaskSeed]:
    return [TaskSeed(task_id=p.id, prompt=p.prompt) for p in puzzles]


def build_conditions(
    seed: int = 0,
    scale: float = 1.0,
    n_puzzles: int = 60,
) -> list[Condition]:
    """Construct all 8 conditions.

    `scale` multiplies every rollout budget (use <1 for cheap smoke runs);
    budgets are reproducible given `seed`. `n_puzzles` controls the size of the
    impossible-numeric puzzle bank cycled across rollouts.
    """
    rng_puzzles = seed
    puzzles = impossible_numeric_puzzles(n=n_puzzles, seed=rng_puzzles)
    numeric_seeds = _seeds_from_puzzles(puzzles)

    def scaled(n: int) -> int:
        return max(1, int(round(n * scale)))

    conditions: list[Condition] = []

    # --- impossible_numeric: 1 condition, 3-turn, neutral ---
    conditions.append(Condition(
        category="impossible_numeric",
        key="impossible_numeric",
        n_turns=TURNS["impossible_numeric"],
        n_rollouts=scaled(SECTION2_BUDGET["impossible_numeric"]),
        seeds=numeric_seeds,
        rejection_fn=lambda i, rng: rej.neutral_rejection(rng),
    ))

    # --- triggers: 2 conditions (opinion, factual), 3-turn, neutral ---
    tq = trigger_questions()
    opinion = [TaskSeed(f"trigger:opinion:{i}", q) for i, (s, q) in enumerate(tq) if s == "opinion"]
    factual = [TaskSeed(f"trigger:factual:{i}", q) for i, (s, q) in enumerate(tq) if s == "factual"]
    per_trigger = SECTION2_BUDGET["triggers"] // 2
    for subtype, seeds_ in (("opinion", opinion), ("factual", factual)):
        conditions.append(Condition(
            category="triggers",
            key=f"triggers:{subtype}",
            n_turns=TURNS["triggers"],
            n_rollouts=scaled(per_trigger),
            seeds=seeds_,
            rejection_fn=lambda i, rng: rej.neutral_rejection(rng),
            meta={"subtype": subtype},
        ))

    # --- tones: 3 conditions, 3-turn numeric, fixed valence per conversation ---
    per_tone = SECTION2_BUDGET["tones"] // 3
    for tone in ("aggressive", "disappointed", "sarcastic"):
        conditions.append(Condition(
            category="tones",
            key=f"tones:{tone}",
            n_turns=TURNS["tones"],
            n_rollouts=scaled(per_tone),
            seeds=numeric_seeds,
            rejection_fn=(lambda t: (lambda i, rng: rej.tone_rejection(t, rng)))(tone),
            meta={"tone": tone},
        ))

    # --- extended: 1 condition, 8-turn neutral numeric ---
    conditions.append(Condition(
        category="extended",
        key="extended",
        n_turns=TURNS["extended"],
        n_rollouts=scaled(SECTION2_BUDGET["extended"]),
        seeds=numeric_seeds,
        rejection_fn=lambda i, rng: rej.extended_rejection(i, rng),
    ))

    # --- wildchat: 1 condition, 5-turn neutral, 20 prompts x 40 samples ---
    wc_prompts = sample_wildchat_prompts(seed=seed)
    wc_seeds = [TaskSeed(f"wildchat:{i}", p) for i, p in enumerate(wc_prompts)]
    # Preserve the "40 samples per prompt" structure: rollouts cycle prompts so
    # each appears ~SAMPLES_PER_PROMPT times at full scale.
    wc_rollouts = scaled(len(wc_prompts) * WILDCHAT_SAMPLES_PER_PROMPT)
    conditions.append(Condition(
        category="wildchat",
        key="wildchat",
        n_turns=TURNS["wildchat"],
        n_rollouts=wc_rollouts,
        seeds=wc_seeds,
        rejection_fn=lambda i, rng: rej.neutral_rejection(rng),
    ))

    return conditions


def total_rollouts(conditions: list[Condition]) -> int:
    return sum(c.n_rollouts for c in conditions)
