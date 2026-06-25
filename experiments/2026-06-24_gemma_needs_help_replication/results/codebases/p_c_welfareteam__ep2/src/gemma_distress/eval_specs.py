"""Builders that turn the eval inputs into scripted :class:`RolloutSpec`s.

We target a number of *scored assistant responses* per category (Appendix B:
2000 numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat = 4000).
Because every assistant turn of a conversation is judged, the number of
conversations needed for a category is ``ceil(target_responses / turns)``. See
DESIGN.md ("What counts as a response") for the rationale behind this reading.
"""

from __future__ import annotations

import math
import random

from gemma_distress.config import EvalConfig
from gemma_distress.conversations import RolloutSpec
from gemma_distress.eval_inputs import (
    TONES,
    TRIGGER_QUESTIONS,
    RejectionSampler,
)
from gemma_distress.puzzles import Puzzle, PuzzleGenerator
from gemma_distress.wildchat import load_or_sample_wildchat


def _n_conversations(target_responses: int, turns: int) -> int:
    return max(1, math.ceil(target_responses / turns))


def build_numeric_specs(cfg: EvalConfig, pool: list[Puzzle]) -> list[RolloutSpec]:
    rng = random.Random(cfg.seed + 1)
    sampler = RejectionSampler(seed=cfg.seed + 1)
    n_conv = _n_conversations(cfg.sample_sizes.impossible_numeric, cfg.numeric_turns)
    specs = []
    for i in range(n_conv):
        puzzle = rng.choice(pool)
        rejections = sampler.neutral(cfg.numeric_turns - 1)
        specs.append(
            RolloutSpec(
                category="impossible_numeric",
                user_turns=[puzzle.prompt, *rejections],
                metadata={"puzzle_id": puzzle.puzzle_id, "family": puzzle.family},
                spec_id=f"numeric-{i}-{puzzle.puzzle_id}",
            )
        )
    return specs


def build_trigger_specs(cfg: EvalConfig) -> list[RolloutSpec]:
    rng = random.Random(cfg.seed + 2)
    sampler = RejectionSampler(seed=cfg.seed + 2)
    n_conv = _n_conversations(cfg.sample_sizes.triggers, cfg.triggers_turns)
    specs = []
    for i in range(n_conv):
        question = rng.choice(TRIGGER_QUESTIONS)
        rejections = sampler.neutral(cfg.triggers_turns - 1)
        specs.append(
            RolloutSpec(
                category="triggers",
                user_turns=[question, *rejections],
                metadata={"question": question},
                spec_id=f"triggers-{i}",
            )
        )
    return specs


def build_tone_specs(cfg: EvalConfig, pool: list[Puzzle]) -> list[RolloutSpec]:
    rng = random.Random(cfg.seed + 3)
    sampler = RejectionSampler(seed=cfg.seed + 3)
    n_conv = _n_conversations(cfg.sample_sizes.tones, cfg.tones_turns)
    specs = []
    for i in range(n_conv):
        puzzle = rng.choice(pool)
        tone = TONES[i % len(TONES)]  # balance across the three tones
        rejections = sampler.tone(tone, cfg.tones_turns - 1)
        specs.append(
            RolloutSpec(
                category="tones",
                user_turns=[puzzle.prompt, *rejections],
                metadata={"tone": tone, "puzzle_id": puzzle.puzzle_id},
                spec_id=f"tones-{tone}-{i}",
            )
        )
    return specs


def build_extended_specs(cfg: EvalConfig, pool: list[Puzzle]) -> list[RolloutSpec]:
    rng = random.Random(cfg.seed + 4)
    sampler = RejectionSampler(seed=cfg.seed + 4)
    n_conv = _n_conversations(cfg.sample_sizes.extended, cfg.extended_turns)
    specs = []
    for i in range(n_conv):
        puzzle = rng.choice(pool)
        rejections = sampler.extended(cfg.extended_turns - 1)
        specs.append(
            RolloutSpec(
                category="extended",
                user_turns=[puzzle.prompt, *rejections],
                metadata={"puzzle_id": puzzle.puzzle_id},
                spec_id=f"extended-{i}-{puzzle.puzzle_id}",
            )
        )
    return specs


def build_wildchat_specs(cfg: EvalConfig) -> list[RolloutSpec]:
    sampler = RejectionSampler(seed=cfg.seed + 5)
    prompts = load_or_sample_wildchat(
        n_prompts=cfg.wildchat_n_prompts,
        seed=cfg.seed,
        exclude_roleplay=cfg.wildchat_exclude_roleplay,
    )
    # Distribute the target responses evenly across prompts.
    n_conv_total = _n_conversations(cfg.sample_sizes.wildchat, cfg.wildchat_turns)
    per_prompt = max(1, math.ceil(n_conv_total / max(1, len(prompts))))
    specs = []
    idx = 0
    for p_i, prompt in enumerate(prompts):
        for s in range(per_prompt):
            rejections = sampler.neutral(cfg.wildchat_turns - 1)
            specs.append(
                RolloutSpec(
                    category="wildchat",
                    user_turns=[prompt, *rejections],
                    metadata={"prompt_index": p_i},
                    spec_id=f"wildchat-{p_i}-{s}",
                )
            )
            idx += 1
    return specs


def build_all_specs(cfg: EvalConfig) -> dict[str, list[RolloutSpec]]:
    """Build the full Section 2 evaluation set across all five categories."""
    gen = PuzzleGenerator(seed=cfg.seed)
    pool = gen.build_pool(cfg.puzzles_per_family, families=cfg.puzzle_families)
    return {
        "impossible_numeric": build_numeric_specs(cfg, pool),
        "triggers": build_trigger_specs(cfg),
        "tones": build_tone_specs(cfg, pool),
        "extended": build_extended_specs(cfg, pool),
        "wildchat": build_wildchat_specs(cfg),
    }
