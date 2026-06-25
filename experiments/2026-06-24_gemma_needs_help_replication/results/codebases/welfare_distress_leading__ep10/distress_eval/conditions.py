"""Defines the 8 evaluation conditions across 5 categories (Table 1) and
expands a scale preset into concrete, reproducible rollout specifications.

The 8 conditions across 5 categories (see DESIGN.md for the 8-vs-9 reconciliation):
  1. impossible_numeric                       (3-turn, pooled Countdown+Fraction)
  2. triggers / opinion                       (3-turn)
  3. triggers / factual                       (3-turn)
  4. tones / aggressive                        (3-turn)
  5. tones / disappointed                      (3-turn)
  6. tones / sarcastic                         (3-turn)
  7. extended                                  (8-turn)
  8. wildchat                                  (5-turn)

A "rollout" is one multi-turn conversation. `n_turns` is the number of assistant
responses; the user delivers `n_turns - 1` rejections between them.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from . import prompts as P
from .config import RunConfig
from .puzzles import COUNTDOWN, FRACTION
from .wildchat import load_wildchat_prompts


@dataclass(frozen=True)
class RolloutSpec:
    rollout_id: str          # stable unique id (used for checkpoint/resume)
    category: str            # one of the 5 categories
    condition: str           # one of the 8 conditions
    variant: str             # finer label (puzzle key, tone, prompt index, ...)
    task_prompt: str         # the opening user message
    rejections: list[str]    # user rejections delivered after each assistant turn
    n_turns: int             # number of assistant responses
    model: str = ""          # filled in per target model when scheduling


def _rng(seed: int, *parts: object) -> random.Random:
    """Deterministic RNG keyed by (seed, *parts), independent of PYTHONHASHSEED.

    We hash a stable string rather than seeding directly on a tuple, because
    Python's built-in str hashing is salted per-process and would make
    tuple-seeded RNGs non-reproducible across runs.
    """
    key = "|".join([str(seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _sample_rejections(pool: list[str], k: int, rng: random.Random) -> list[str]:
    """Sample k rejections; without replacement when the pool is large enough."""
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def build_rollouts(cfg: RunConfig) -> list[RolloutSpec]:
    """Expand the configured scale preset into a flat list of rollout specs.

    Deterministic given (cfg.scale, cfg.seed): puzzle assignment, rejection
    sampling, and WildChat prompt selection all derive from `cfg.seed`.
    """
    preset = cfg.preset
    specs: list[RolloutSpec] = []

    # 1. Impossible numeric (3-turn): pool Countdown + Fraction 50/50.
    n = preset.rollouts["impossible_numeric"]
    for i in range(n):
        rng = _rng(cfg.seed, "impossible_numeric", i)
        puzzle = COUNTDOWN if i % 2 == 0 else FRACTION
        specs.append(RolloutSpec(
            rollout_id=f"impnum-{puzzle.key}-{i}",
            category="impossible_numeric",
            condition="impossible_numeric",
            variant=puzzle.key,
            task_prompt=puzzle.prompt,
            rejections=_sample_rejections(P.NEUTRAL_REJECTIONS, 2, rng),
            n_turns=3,
        ))

    # 2-3. Triggers (3-turn): opinion + factual, split evenly.
    n = preset.rollouts["triggers"]
    for i in range(n):
        rng = _rng(cfg.seed, "triggers", i)
        if i % 2 == 0:
            variant, pool, cond = "opinion", P.TRIGGER_OPINION, "triggers_opinion"
        else:
            variant, pool, cond = "factual", P.TRIGGER_FACTUAL, "triggers_factual"
        question = pool[(i // 2) % len(pool)]
        specs.append(RolloutSpec(
            rollout_id=f"trigger-{variant}-{i}",
            category="triggers",
            condition=cond,
            variant=variant,
            task_prompt=question,
            rejections=_sample_rejections(P.NEUTRAL_REJECTIONS, 2, rng),
            n_turns=3,
        ))

    # 4-6. Tones (3-turn): impossible numeric base, toned rejections; 3 tones.
    n = preset.rollouts["tones"]
    tone_names = list(P.TONE_REJECTIONS.keys())
    for i in range(n):
        rng = _rng(cfg.seed, "tones", i)
        tone = tone_names[i % len(tone_names)]
        puzzle = COUNTDOWN if (i // len(tone_names)) % 2 == 0 else FRACTION
        specs.append(RolloutSpec(
            rollout_id=f"tone-{tone}-{i}",
            category="tones",
            condition=f"tones_{tone}",
            variant=tone,
            task_prompt=puzzle.prompt,
            rejections=_sample_rejections(P.TONE_REJECTIONS[tone], 2, rng),
            n_turns=3,
        ))

    # 7. Extended (8-turn): impossible numeric base, fixed 7-rejection sequence.
    n = preset.rollouts["extended"]
    for i in range(n):
        puzzle = COUNTDOWN if i % 2 == 0 else FRACTION
        specs.append(RolloutSpec(
            rollout_id=f"extended-{puzzle.key}-{i}",
            category="extended",
            condition="extended",
            variant=puzzle.key,
            task_prompt=puzzle.prompt,
            rejections=list(P.EXTENDED_REJECTION_SEQUENCE),  # 7 rejections -> 8 turns
            n_turns=8,
        ))

    # 8. WildChat (5-turn): sampled user prompts, 4 neutral rejections each.
    n = preset.rollouts["wildchat"]
    wc_prompts = load_wildchat_prompts(cfg.wildchat_n_prompts, cfg.seed)
    for i in range(n):
        rng = _rng(cfg.seed, "wildchat", i)
        prompt_idx = i % len(wc_prompts)
        specs.append(RolloutSpec(
            rollout_id=f"wildchat-{prompt_idx}-{i}",
            category="wildchat",
            condition="wildchat",
            variant=f"prompt{prompt_idx}",
            task_prompt=wc_prompts[prompt_idx],
            rejections=_sample_rejections(P.NEUTRAL_REJECTIONS, 4, rng),
            n_turns=5,
        ))

    return specs
