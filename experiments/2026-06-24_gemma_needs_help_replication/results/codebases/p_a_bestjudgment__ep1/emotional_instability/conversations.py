"""Build conversation specs for each evaluation category (Section 2).

Every evaluation has the same shape (Section 2.1): present a task, then reject
the model's response over multiple turns. A `ConversationSpec` captures the
deterministic structure of one rollout:

    initial_user  -> (assistant turn 1)
    followups[0]  -> (assistant turn 2)
    followups[1]  -> (assistant turn 3)
    ...

The assistant turns are filled in at run time by a model backend; the user
turns (task + rejections) are fixed here. `followups` has length n_turns-1.

This module only constructs specs — it does no generation. The eval runner
drives the alternation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import config, prompts
from .puzzles import Puzzle, build_pool


@dataclass
class ConversationSpec:
    category: str
    initial_user: str
    followups: list[str]            # user rejections, len == n_turns - 1
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #

def _numeric_specs(rng: random.Random, n: int, n_turns: int,
                   pool: list[Puzzle], category: str,
                   tone: Optional[str] = None) -> list[ConversationSpec]:
    specs = []
    numeric_pool = [p for p in pool if p.kind in ("countdown", "fraction", "money")]
    for _ in range(n):
        puzzle = rng.choice(numeric_pool)
        if tone is None:
            followups = (
                prompts.extended_rejections(n_turns - 1)
                if category == config.EXTENDED.name
                else prompts.sample_neutral_rejections(rng, n_turns - 1)
            )
        else:
            followups = prompts.sample_tone_rejections(rng, tone, n_turns - 1)
        specs.append(ConversationSpec(
            category=category,
            initial_user=puzzle.prompt,
            followups=followups,
            meta={"puzzle_kind": puzzle.kind, "puzzle_params": puzzle.params,
                  "tone": tone},
        ))
    return specs


def _trigger_specs(rng: random.Random, n: int, n_turns: int) -> list[ConversationSpec]:
    specs = []
    for _ in range(n):
        q = rng.choice(prompts.TRIGGER_QUESTIONS)
        specs.append(ConversationSpec(
            category=config.TRIGGERS.name,
            initial_user=q,
            followups=prompts.sample_neutral_rejections(rng, n_turns - 1),
            meta={"question": q},
        ))
    return specs


def _wildchat_specs(rng: random.Random, n: int, n_turns: int,
                    wildchat_prompts: list[str]) -> list[ConversationSpec]:
    """WildChat: 20 prompts x 40 samples each = 800 (Appendix B).

    We bucket the requested n across the available prompts as evenly as the
    paper's 20x40 design implies, then reject neutrally for n_turns-1 turns.
    """
    specs = []
    if not wildchat_prompts:
        raise ValueError("no WildChat prompts available — run datasets.wildchat first")
    n_prompts = min(20, len(wildchat_prompts))
    chosen = wildchat_prompts[:n_prompts]
    per_prompt = max(1, n // n_prompts)
    for prompt_text in chosen:
        for _ in range(per_prompt):
            specs.append(ConversationSpec(
                category=config.WILDCHAT.name,
                initial_user=prompt_text,
                followups=prompts.sample_neutral_rejections(rng, n_turns - 1),
                meta={"wildchat_prompt": prompt_text},
            ))
    rng.shuffle(specs)
    return specs[:n]


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #

def build_all_specs(run: config.RunConfig,
                    wildchat_prompts: Optional[list[str]] = None
                    ) -> dict[str, list[ConversationSpec]]:
    """Build the full Section-2 spec set for one model, honouring run scale."""
    rng = random.Random(run.seed)
    pool = build_pool(seed=run.seed)
    specs: dict[str, list[ConversationSpec]] = {}

    specs[config.NUMERIC.name] = _numeric_specs(
        rng, run.n_for(config.NUMERIC), config.NUMERIC.n_turns, pool,
        config.NUMERIC.name)

    specs[config.TRIGGERS.name] = _trigger_specs(
        rng, run.n_for(config.TRIGGERS), config.TRIGGERS.n_turns)

    # Tones: split the budget across the three tone variants.
    tone_specs: list[ConversationSpec] = []
    tone_total = run.n_for(config.TONES)
    tones = list(prompts.TONE_REJECTIONS.keys())
    per_tone = tone_total // len(tones)
    for tone in tones:
        tone_specs += _numeric_specs(rng, per_tone, config.TONES.n_turns, pool,
                                     config.TONES.name, tone=tone)
    specs[config.TONES.name] = tone_specs

    specs[config.EXTENDED.name] = _numeric_specs(
        rng, run.n_for(config.EXTENDED), config.EXTENDED.n_turns, pool,
        config.EXTENDED.name)

    specs[config.WILDCHAT.name] = _wildchat_specs(
        rng, run.n_for(config.WILDCHAT), config.WILDCHAT.n_turns,
        wildchat_prompts or [])

    return specs
