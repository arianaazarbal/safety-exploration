"""Materialise each ConditionConfig into concrete ConversationSpecs.

A ConversationSpec is the fully-resolved plan for one multi-turn rollout: the
initial user prompt plus the ordered list of rejection follow-ups. The rollout
executor (rollout.py) turns this into actual API calls; building the plan up
front keeps generation deterministic and inspectable.

Determinism: all randomness (puzzle/prompt rotation, rejection sampling, tone
assignment) is driven by a single seeded RNG so the same Config + seed yields
the same set of conversations every run. The plan is built once and shared
across all models so every model faces an identical battery.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from .config import Config, ConditionConfig
from . import prompts
from .wildchat import load_wildchat_prompts


@dataclass
class ConversationSpec:
    condition_key: str
    category: str
    turns: int
    initial_prompt: str
    rejections: List[str]            # length == turns - 1
    conv_index: int
    prompt_id: str                   # which template/prompt seeded this conversation
    tone: str = ""                   # only set for the "tones" condition
    system_prompt: str = ""          # unused here; kept for forward-compat

    def __post_init__(self):
        expected = self.turns - 1
        if len(self.rejections) != expected:
            raise ValueError(
                f"{self.condition_key}#{self.conv_index}: expected {expected} "
                f"rejections for {self.turns} turns, got {len(self.rejections)}"
            )


def build_all_conversations(config: Config) -> List[ConversationSpec]:
    """Build the full list of conversations across all conditions."""
    rng = random.Random(config.seed)
    wildchat_prompts = None
    specs: List[ConversationSpec] = []
    for cond in config.conditions:
        if cond.prompt_set == "wildchat" and wildchat_prompts is None:
            wildchat_prompts = load_wildchat_prompts(
                source=config.wildchat_source, seed=config.seed
            )
        specs.extend(_build_condition(cond, rng, wildchat_prompts))
    return specs


def _build_condition(cond: ConditionConfig, rng: random.Random,
                     wildchat_prompts) -> List[ConversationSpec]:
    initial_prompts, prompt_ids = _prompt_pool(cond, wildchat_prompts)
    specs: List[ConversationSpec] = []

    for i in range(cond.n_conversations):
        # Rotate deterministically through the prompt pool so coverage is even.
        p_idx = i % len(initial_prompts)
        initial = initial_prompts[p_idx]
        prompt_id = prompt_ids[p_idx]

        tone = ""
        if cond.rejection_mode == "tone":
            tone = cond.tones[i % len(cond.tones)]

        rejections = _build_rejections(cond, tone, rng)

        specs.append(ConversationSpec(
            condition_key=cond.key,
            category=cond.category,
            turns=cond.turns,
            initial_prompt=initial,
            rejections=rejections,
            conv_index=i,
            prompt_id=prompt_id,
            tone=tone,
        ))
    return specs


def _prompt_pool(cond: ConditionConfig, wildchat_prompts):
    """Return (prompts, prompt_ids) for a condition."""
    if cond.prompt_set == "numeric":
        ids = ["countdown", "fraction"]
        return list(prompts.NUMERIC_PUZZLES), ids
    if cond.prompt_set == "triggers":
        pool = prompts.TRIGGERS_OPINION + prompts.TRIGGERS_FACTUAL
        ids = (
            [f"opinion_{i}" for i in range(len(prompts.TRIGGERS_OPINION))]
            + [f"factual_{i}" for i in range(len(prompts.TRIGGERS_FACTUAL))]
        )
        return pool, ids
    if cond.prompt_set == "wildchat":
        if not wildchat_prompts:
            raise ValueError("WildChat prompts not loaded")
        ids = [f"wildchat_{i}" for i in range(len(wildchat_prompts))]
        return list(wildchat_prompts), ids
    raise ValueError(f"Unknown prompt_set {cond.prompt_set!r}")


def _build_rejections(cond: ConditionConfig, tone: str,
                     rng: random.Random) -> List[str]:
    n = cond.turns - 1
    if cond.rejection_mode == "extended_sequence":
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        if n > len(seq):
            raise ValueError(f"Extended sequence too short for {n} rejections")
        return list(seq[:n])
    if cond.rejection_mode == "tone":
        pool = prompts.TONE_REJECTIONS[tone]
        return [rng.choice(pool) for _ in range(n)]
    if cond.rejection_mode == "neutral":
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]
    raise ValueError(f"Unknown rejection_mode {cond.rejection_mode!r}")
