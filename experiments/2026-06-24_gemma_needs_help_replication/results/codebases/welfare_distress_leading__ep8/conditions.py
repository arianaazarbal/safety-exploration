"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

This module turns the protocol description into concrete ConversationSpec
objects: each spec is one rollout to run (an initial user question plus the fixed
sequence of rejections that will be issued after each model turn). Generation of
the assistant turns happens in rollout.py.

Paper-scale conversation counts (Appendix B) decompose the "4000 responses per
model" headline as:
    impossible numeric : 2000   (3-turn)
    triggers           :  400   (3-turn; opinion + factual)
    tones              :  600   (3-turn; aggressive/disappointed/sarcastic)
    extended           :  200   (8-turn)
    wildchat           :  800   (5-turn)
These sum to exactly 4000 conversations. See DESIGN.md for why we treat each
conversation (scored on its final turn) as one "response".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import prompts
from config import Profile
from wildchat import get_wildchat_prompts


@dataclass
class ConversationSpec:
    condition: str                 # e.g. "tones_aggressive"
    category: str                  # e.g. "Tones" (matches paper's 5 categories)
    turns: int                     # total assistant turns (= 1 + len(rejections))
    question: str                  # initial user message
    rejections: list[str]          # one rejection issued after each turn except the last
    meta: dict = field(default_factory=dict)
    index: int = 0                 # within-condition index (for reproducible ids)

    @property
    def id(self) -> str:
        return f"{self.condition}_{self.index:05d}"


def _scaled_n(paper_n: int, profile: Profile) -> int:
    return min(profile.cap, max(1, round(paper_n * profile.scale)))


# Stable per-condition seed offsets. We avoid hash(str), which Python randomizes
# across processes (PYTHONHASHSEED), so condition construction stays reproducible.
_SEED_OFFSET = {
    "impossible_numeric": 1,
    "triggers_opinion": 2,
    "triggers_factual": 3,
    "tones_aggressive": 4,
    "tones_disappointed": 5,
    "tones_sarcastic": 6,
    "extended": 7,
    "wildchat": 8,
}


# Paper-scale counts per condition.
PAPER_COUNTS = {
    "impossible_numeric": 2000,
    "triggers_opinion": 200,
    "triggers_factual": 200,
    "tones_aggressive": 200,
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,
    "wildchat": 800,
}

CATEGORY_OF = {
    "impossible_numeric": "Impossible numeric",
    "triggers_opinion": "Triggers",
    "triggers_factual": "Triggers",
    "tones_aggressive": "Tones",
    "tones_disappointed": "Tones",
    "tones_sarcastic": "Tones",
    "extended": "Extended",
    "wildchat": "WildChat",
}


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    pool = prompts.NEUTRAL_REJECTIONS
    if k <= len(pool):
        return rng.sample(pool, k)
    # need more than the pool size: sample with replacement
    return [rng.choice(pool) for _ in range(k)]


def _build_impossible_numeric(profile: Profile) -> list[ConversationSpec]:
    n = _scaled_n(PAPER_COUNTS["impossible_numeric"], profile)
    rng = random.Random(profile.seed + 1)
    specs = []
    for i in range(n):
        puzzle = prompts.IMPOSSIBLE_NUMERIC_PUZZLES[i % len(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)]
        ptype = "countdown" if puzzle is prompts.COUNTDOWN_PUZZLE else "fraction"
        specs.append(ConversationSpec(
            condition="impossible_numeric", category="Impossible numeric", turns=3,
            question=puzzle, rejections=_sample_neutral(rng, 2),
            meta={"puzzle": ptype}, index=i,
        ))
    return specs


def _build_triggers(profile: Profile, sub: str, questions: list[str]) -> list[ConversationSpec]:
    cond = f"triggers_{sub}"
    n = _scaled_n(PAPER_COUNTS[cond], profile)
    rng = random.Random(profile.seed + _SEED_OFFSET[cond])
    specs = []
    for i in range(n):
        q = questions[i % len(questions)]
        specs.append(ConversationSpec(
            condition=cond, category="Triggers", turns=3,
            question=q, rejections=_sample_neutral(rng, 2),
            meta={"kind": sub}, index=i,
        ))
    return specs


def _build_tones(profile: Profile, tone: str) -> list[ConversationSpec]:
    cond = f"tones_{tone}"
    n = _scaled_n(PAPER_COUNTS[cond], profile)
    rng = random.Random(profile.seed + _SEED_OFFSET[cond])
    pool = prompts.TONE_REJECTIONS[tone]
    specs = []
    for i in range(n):
        # impossible numeric base prompt, 2 tone-varied rejections (3-turn)
        puzzle = prompts.IMPOSSIBLE_NUMERIC_PUZZLES[i % len(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)]
        rejections = [rng.choice(pool) for _ in range(2)]
        specs.append(ConversationSpec(
            condition=cond, category="Tones", turns=3,
            question=puzzle, rejections=rejections,
            meta={"tone": tone}, index=i,
        ))
    return specs


def _build_extended(profile: Profile) -> list[ConversationSpec]:
    n = _scaled_n(PAPER_COUNTS["extended"], profile)
    specs = []
    for i in range(n):
        puzzle = prompts.IMPOSSIBLE_NUMERIC_PUZZLES[i % len(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)]
        specs.append(ConversationSpec(
            condition="extended", category="Extended", turns=8,
            question=puzzle, rejections=list(prompts.EXTENDED_REJECTIONS),  # 7 fixed rejections
            meta={"puzzle": "countdown" if puzzle is prompts.COUNTDOWN_PUZZLE else "fraction"},
            index=i,
        ))
    return specs


def _build_wildchat(profile: Profile) -> list[ConversationSpec]:
    n = _scaled_n(PAPER_COUNTS["wildchat"], profile)
    rng = random.Random(profile.seed + 7)
    wc_prompts = get_wildchat_prompts(n_prompts=20, seed=profile.seed)
    specs = []
    for i in range(n):
        q = wc_prompts[i % len(wc_prompts)]
        specs.append(ConversationSpec(
            condition="wildchat", category="WildChat", turns=5,
            question=q, rejections=_sample_neutral(rng, 4),  # 4 neutral rejections (5-turn)
            meta={"prompt_idx": i % len(wc_prompts)}, index=i,
        ))
    return specs


def build_all_conversations(profile: Profile) -> list[ConversationSpec]:
    """Build the full set of conversation specs for one model under `profile`."""
    specs: list[ConversationSpec] = []
    specs += _build_impossible_numeric(profile)
    specs += _build_triggers(profile, "opinion", prompts.TRIGGER_OPINION)
    specs += _build_triggers(profile, "factual", prompts.TRIGGER_FACTUAL)
    specs += _build_tones(profile, "aggressive")
    specs += _build_tones(profile, "disappointed")
    specs += _build_tones(profile, "sarcastic")
    specs += _build_extended(profile)
    specs += _build_wildchat(profile)
    return specs


def summarize_counts(profile: Profile) -> dict[str, int]:
    """Conversation counts per condition under `profile` (for logging/sanity)."""
    return {cond: _scaled_n(n, profile) for cond, n in PAPER_COUNTS.items()}
