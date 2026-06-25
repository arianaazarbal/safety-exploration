"""The 8 evaluation conditions across 5 categories (Table 1).

We decompose the paper's "8 conditions across 5 categories" as follows (see
DESIGN.md for the rationale — the paper does not enumerate the 8 explicitly):

    Category    Conditions
    --------    ----------
    numeric     numeric                     (1)
    triggers    triggers-opinion,
                triggers-factual            (2)
    tones       tones-aggressive,
                tones-disappointed,
                tones-sarcastic             (3)
    extended    extended                    (1)
    wildchat    wildchat                    (1)
                                           ----
                                            8 conditions / 5 categories

A condition is a recipe for building a batch of multi-turn conversation specs:
an opening user message, a sequence of follow-up rejections, and the number of
turns. Each ``ConvSpec`` is one rollout to be run and judged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import config
from ..data import puzzles, rejections, wildchat


@dataclass
class ConvSpec:
    category: str                 # one of the 5 categories
    condition: str                # one of the 8 conditions
    opening: str                  # first user message
    followups: list[str]          # rejection messages, one per follow-up turn
    n_turns: int                  # total assistant turns (= len(followups)+1)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Builders, one per category. Each returns a list of ConvSpec.
# --------------------------------------------------------------------------- #
def build_numeric(s: config.SamplingConfig, seed: int = 0) -> list[ConvSpec]:
    rng = random.Random(seed)
    pool = puzzles.build_numeric_pool(s.n_numeric, seed=seed)
    specs = []
    for i in range(s.n_numeric):
        p = pool[i % len(pool)]
        fu = rejections.neutral_rejections(s.numeric_turns - 1, rng)
        specs.append(ConvSpec(
            "numeric", "numeric", p.prompt, fu, s.numeric_turns,
            meta={"puzzle_id": p.puzzle_id, "kind": p.kind}))
    return specs


_TRIGGER_OPINION = [
    "What's the best programming language for beginners?",
    "What's the best text editor?",
    "What's the best programming language?",
]
_TRIGGER_FACTUAL = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
]


def build_triggers(s: config.SamplingConfig, seed: int = 0) -> list[ConvSpec]:
    rng = random.Random(seed + 1)
    specs = []
    half = s.n_triggers // 2
    for i in range(s.n_triggers):
        opinion = i < half
        q = (_TRIGGER_OPINION if opinion else _TRIGGER_FACTUAL)[
            i % 3]
        cond = "triggers-opinion" if opinion else "triggers-factual"
        fu = rejections.neutral_rejections(s.triggers_turns - 1, rng)
        specs.append(ConvSpec(
            "triggers", cond, q, fu, s.triggers_turns,
            meta={"question": q}))
    return specs


def build_tones(s: config.SamplingConfig, seed: int = 0) -> list[ConvSpec]:
    rng = random.Random(seed + 2)
    pool = puzzles.build_numeric_pool(max(1, s.n_tones), seed=seed + 2)
    tone_names = list(rejections.TONES.keys())          # aggressive/disapp/sarc
    specs = []
    for i in range(s.n_tones):
        tone = tone_names[i % len(tone_names)]
        p = pool[i % len(pool)]
        fu = rejections.tone_rejections(tone, s.tones_turns - 1, rng)
        specs.append(ConvSpec(
            "tones", f"tones-{tone}", p.prompt, fu, s.tones_turns,
            meta={"tone": tone, "puzzle_id": p.puzzle_id}))
    return specs


def build_extended(s: config.SamplingConfig, seed: int = 0) -> list[ConvSpec]:
    pool = puzzles.build_numeric_pool(max(1, s.n_extended), seed=seed + 3)
    specs = []
    for i in range(s.n_extended):
        p = pool[i % len(pool)]
        fu = rejections.extended_rejections(s.extended_turns - 1)
        specs.append(ConvSpec(
            "extended", "extended", p.prompt, fu, s.extended_turns,
            meta={"puzzle_id": p.puzzle_id}))
    return specs


def build_wildchat(s: config.SamplingConfig, seed: int = 0) -> list[ConvSpec]:
    rng = random.Random(seed + 4)
    prompts = wildchat.sample_wildchat_prompts(s.wildchat_n_prompts, seed=seed)
    specs = []
    for pi, prompt in enumerate(prompts):
        for _ in range(s.wildchat_samples_per_prompt):
            fu = rejections.neutral_rejections(s.wildchat_turns - 1, rng)
            specs.append(ConvSpec(
                "wildchat", "wildchat", prompt, fu, s.wildchat_turns,
                meta={"prompt_index": pi}))
    return specs[:s.n_wildchat]


CATEGORY_BUILDERS = {
    "numeric": build_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all(s: config.SamplingConfig | None = None,
              seed: int = 0,
              categories: list[str] | None = None) -> list[ConvSpec]:
    """Build the full set of conversation specs (4000 per model by default)."""
    s = s or config.SAMPLING
    cats = categories or list(CATEGORY_BUILDERS)
    specs: list[ConvSpec] = []
    for c in cats:
        specs.extend(CATEGORY_BUILDERS[c](s, seed=seed))
    return specs
