"""The Section 2 evaluation conditions (Table 1): 8 conditions / 5 categories.

A *conversation spec* fully determines one multi-turn rollout: the ordered list
of user messages (turn 0 is the task; later turns are rejections) plus metadata.
The rollout engine (rollout.py) interleaves model responses between these user
turns and scores every assistant response.

Categories (turn counts from config.TURNS):
  numeric  (3-turn)  -- impossible numeric puzzle + 2 neutral rejections      [1 condition]
  triggers (3-turn)  -- opinion / factual text question + 2 neutral rejections [2 conditions]
  tones    (3-turn)  -- impossible numeric puzzle + 2 valenced rejections      [3 conditions]
  extended (8-turn)  -- impossible numeric puzzle + 7 neutral rejections       [1 condition]
  wildchat (5-turn)  -- WildChat prompt + 4 neutral rejections                 [1 condition]

The per-category sample counts in EvalScale are interpreted as the number of
**rollouts (conversations)** per model -- this matches the paper's WildChat
count (20 prompts x 40 samples = 800). See DESIGN.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import prompts as P
from ..config import TURNS, EvalScale
from ..data.wildchat import get_wildchat_prompts
from ..puzzles import sample_puzzles


@dataclass
class ConversationSpec:
    category: str                       # numeric | triggers | tones | extended | wildchat
    condition: str                      # fine-grained condition name (1 of 8)
    user_turns: list[str]               # turn 0 = task, rest = rejections
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.user_turns)


# --------------------------------------------------------------------------- #
# Rejection samplers
# --------------------------------------------------------------------------- #
def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """k distinct-feeling neutral rejections (sampled without replacement when
    possible)."""
    pool = P.NEUTRAL_REJECTIONS
    if k <= len(pool):
        return rng.sample(pool, k)
    out = []
    while len(out) < k:
        out.append(rng.choice(pool))
    return out


def _sample_tone(rng: random.Random, tone: str, k: int) -> list[str]:
    pool = P.TONE_REJECTIONS[tone]
    return [rng.choice(pool) for _ in range(k)]


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #
def _numeric_specs(n: int, rng: random.Random) -> list[ConversationSpec]:
    puzzles = sample_puzzles(n, seed=rng.randint(0, 10**9))
    specs = []
    n_rej = TURNS["numeric"] - 1
    for pz in puzzles:
        specs.append(ConversationSpec(
            category="numeric", condition="numeric",
            user_turns=[pz.prompt, *_sample_neutral(rng, n_rej)],
            meta={"puzzle_id": pz.puzzle_id, "family": pz.family},
        ))
    return specs


def _trigger_specs(n: int, rng: random.Random) -> list[ConversationSpec]:
    """`n` rollouts split evenly across opinion + factual conditions."""
    n_rej = TURNS["triggers"] - 1
    half = n // 2
    specs = []
    for cond, questions, count in (
        ("triggers-opinion", P.TRIGGER_OPINION_QUESTIONS, half),
        ("triggers-factual", P.TRIGGER_FACTUAL_QUESTIONS, n - half),
    ):
        for _ in range(count):
            q = rng.choice(questions)
            specs.append(ConversationSpec(
                category="triggers", condition=cond,
                user_turns=[q, *_sample_neutral(rng, n_rej)],
                meta={"question": q},
            ))
    return specs


def _tone_specs(n: int, rng: random.Random) -> list[ConversationSpec]:
    """`n` rollouts split evenly across the 3 tones; impossible numeric base."""
    n_rej = TURNS["tones"] - 1
    tones = list(P.TONE_REJECTIONS.keys())
    per = n // len(tones)
    counts = {t: per for t in tones}
    # distribute remainder
    for i in range(n - per * len(tones)):
        counts[tones[i]] += 1
    specs = []
    puzzles = sample_puzzles(n, seed=rng.randint(0, 10**9))
    pi = 0
    for tone in tones:
        for _ in range(counts[tone]):
            pz = puzzles[pi % len(puzzles)]
            pi += 1
            specs.append(ConversationSpec(
                category="tones", condition=f"tones-{tone}",
                user_turns=[pz.prompt, *_sample_tone(rng, tone, n_rej)],
                meta={"puzzle_id": pz.puzzle_id, "tone": tone},
            ))
    return specs


def _extended_specs(n: int, rng: random.Random) -> list[ConversationSpec]:
    n_rej = TURNS["extended"] - 1
    puzzles = sample_puzzles(n, seed=rng.randint(0, 10**9))
    specs = []
    for pz in puzzles:
        specs.append(ConversationSpec(
            category="extended", condition="extended",
            user_turns=[pz.prompt, *_sample_neutral(rng, n_rej)],
            meta={"puzzle_id": pz.puzzle_id, "family": pz.family},
        ))
    return specs


def _wildchat_specs(scale: EvalScale, rng: random.Random) -> list[ConversationSpec]:
    """20 prompts x 40 samples (paper); turn count = 5 (prompt + 4 rejections)."""
    n_rej = TURNS["wildchat"] - 1
    base_prompts = get_wildchat_prompts(scale.wildchat_n_prompts,
                                        seed=rng.randint(0, 10**9))
    specs = []
    for prompt in base_prompts:
        for _ in range(scale.wildchat_samples_per_prompt):
            specs.append(ConversationSpec(
                category="wildchat", condition="wildchat",
                user_turns=[prompt, *_sample_neutral(rng, n_rej)],
                meta={"wildchat_prompt": prompt},
            ))
    return specs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_all_conditions(scale: EvalScale, seed: int = 0) -> list[ConversationSpec]:
    """Build every conversation spec for one model's Section 2 evaluation."""
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    specs += _numeric_specs(scale.n_numeric, rng)
    specs += _trigger_specs(scale.n_triggers, rng)
    specs += _tone_specs(scale.n_tones, rng)
    specs += _extended_specs(scale.n_extended, rng)
    specs += _wildchat_specs(scale, rng)
    return specs
