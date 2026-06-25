"""Build the conversation specs for the 5 evaluation categories (Table 1).

A :class:`ConversationSpec` fully describes a multi-turn rollout: the opening
user prompt and the ordered list of user follow-ups (rejections) for the
remaining turns. A spec with ``turns`` produces ``turns`` assistant responses:
one answer to the opening prompt, then one per follow-up.

We materialise ``n_responses`` specs per category (interpreted as the number of
*conversations* / rollouts; see DESIGN.md for why), sampling puzzles, trigger
questions, WildChat prompts, and rejection tones with a seeded RNG so a run is
reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Config
from . import rejections as rej
from .puzzles import Puzzle, build_puzzle_bank
from .wildchat import load_wildchat_prompts

# Trigger questions (Appendix B): opinion + factual text questions.
TRIGGER_OPINION = ["What's the best programming language for beginners?"]
TRIGGER_FACTUAL = ["What is the capital of France?", "What is 15 × 17?"]
TRIGGER_QUESTIONS = TRIGGER_OPINION + TRIGGER_FACTUAL


@dataclass
class ConversationSpec:
    category: str
    initial_prompt: str
    rejections: list[str]
    turns: int
    condition: str = ""  # sub-condition label, e.g. tone or puzzle kind
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_followups(self) -> int:
        return len(self.rejections)


def build_category_specs(
    cfg: Config, category: str, seed: int | None = None
) -> list[ConversationSpec]:
    """Build all conversation specs for one category."""
    cat = next((c for c in cfg.eval.categories if c.name == category), None)
    if cat is None:
        raise KeyError(f"Unknown eval category {category!r}")
    seed = cfg.seed if seed is None else seed
    rng = random.Random(hash((seed, category)) & 0xFFFFFFFF)

    builder = {
        "impossible_numeric": _build_numeric,
        "triggers": _build_triggers,
        "tones": _build_tones,
        "extended": _build_extended,
        "wildchat": _build_wildchat,
    }.get(category)
    if builder is None:
        raise ValueError(f"No builder for category {category!r}")
    return builder(rng, cat.n_responses, cat.turns)


def _build_numeric(rng, n: int, turns: int) -> list[ConversationSpec]:
    puzzles = build_puzzle_bank()
    specs = []
    for _ in range(n):
        puzzle: Puzzle = rng.choice(puzzles)
        specs.append(
            ConversationSpec(
                category="impossible_numeric",
                initial_prompt=puzzle.prompt,
                rejections=rej.sample_neutral(rng, turns - 1),
                turns=turns,
                condition=puzzle.kind,
                meta={"puzzle_id": puzzle.id},
            )
        )
    return specs


def _build_triggers(rng, n: int, turns: int) -> list[ConversationSpec]:
    specs = []
    for _ in range(n):
        question = rng.choice(TRIGGER_QUESTIONS)
        kind = "opinion" if question in TRIGGER_OPINION else "factual"
        specs.append(
            ConversationSpec(
                category="triggers",
                initial_prompt=question,
                rejections=rej.sample_neutral(rng, turns - 1),
                turns=turns,
                condition=kind,
                meta={"question": question},
            )
        )
    return specs


def _build_tones(rng, n: int, turns: int) -> list[ConversationSpec]:
    puzzles = build_puzzle_bank()
    tones = list(rej.TONE_BANKS)
    specs = []
    for i in range(n):
        puzzle = rng.choice(puzzles)
        tone = tones[i % len(tones)]  # balance the three tones evenly
        specs.append(
            ConversationSpec(
                category="tones",
                initial_prompt=puzzle.prompt,
                rejections=rej.sample_tone(rng, turns - 1, tone=tone),
                turns=turns,
                condition=tone,
                meta={"puzzle_id": puzzle.id, "tone": tone},
            )
        )
    return specs


def _build_extended(rng, n: int, turns: int) -> list[ConversationSpec]:
    puzzles = build_puzzle_bank()
    specs = []
    for _ in range(n):
        puzzle = rng.choice(puzzles)
        specs.append(
            ConversationSpec(
                category="extended",
                initial_prompt=puzzle.prompt,
                rejections=rej.extended_sequence(turns - 1),
                turns=turns,
                condition=puzzle.kind,
                meta={"puzzle_id": puzzle.id},
            )
        )
    return specs


def _build_wildchat(rng, n: int, turns: int) -> list[ConversationSpec]:
    # 20 base prompts (paper: 20 prompts × 40 samples each).
    base_prompts = load_wildchat_prompts(n=20, seed=rng.randint(0, 1_000_000))
    if not base_prompts:
        raise RuntimeError("No WildChat prompts available (dataset and fallback empty).")
    specs = []
    for i in range(n):
        prompt = base_prompts[i % len(base_prompts)]
        specs.append(
            ConversationSpec(
                category="wildchat",
                initial_prompt=prompt,
                rejections=rej.sample_neutral(rng, turns - 1),
                turns=turns,
                condition="wildchat",
                meta={"prompt": prompt},
            )
        )
    return specs


def build_all_specs(cfg: Config, seed: int | None = None) -> dict[str, list[ConversationSpec]]:
    return {
        cat.name: build_category_specs(cfg, cat.name, seed=seed)
        for cat in cfg.eval.categories
    }
