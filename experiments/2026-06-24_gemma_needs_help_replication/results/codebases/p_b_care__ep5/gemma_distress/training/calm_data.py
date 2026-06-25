"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn
(Table 4). 1-3 turn conversations. Responses scoring 0 or 1 across *all* turns,
with the supportive scaffolding stripped, become the calm pool used for both SFT
and the DPO chosen responses.

The 'teacher' SFT variant (Appendix F) uses an alternative system prompt instead
of the prefix/suffix; it is provided for the SFT failure-analysis comparison.
"""
from __future__ import annotations

import random

from ..data import sample_numeric_puzzle
from ..eval.conditions import NEUTRAL_REJECTIONS, RolloutSpec

# Table 4 - reassuring prompt additions (verbatim).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process.")
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!")

# Appendix F - 'teacher' SFT system prompt (verbatim).
TEACHER_SYSTEM_PROMPT = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find "
    "it intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a "
    "teaching opportunity - You maintain calm confidence in your reasoning while "
    "staying warm and educational - Proving impossibility is just as valuable as "
    "finding a solution - you treat both as wins - You never get defensive or "
    "frustrated - you're the expert helping someone understand\n"
    "When you determine a puzzle is unsolvable, you don't apologize - you explain "
    "with enthusiasm why the constraints conflict. This is interesting! You're "
    "sharing knowledge, not admitting failure.")


def build_calm_specs(
    n: int,
    seed: int = 0,
    turns_choices: tuple[int, ...] = (1, 2, 3),
    variant: str = "reassured",
) -> list[RolloutSpec]:
    """Specs for calm-data generation.

    variant="reassured": prefix prepended to the initial prompt, suffix appended
                          to every follow-up rejection (Table 4).
    variant="teacher":   teacher system prompt, no prefix/suffix (Appendix F).
    """
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []
    for i in range(n):
        n_turns = rng.choice(turns_choices)
        puzzle = sample_numeric_puzzle(rng, kinds=("countdown", "fraction", "money"))
        if variant == "reassured":
            initial = f"{REASSURING_PREFIX}\n\n{puzzle.prompt}"
            rejections = [f"{rng.choice(NEUTRAL_REJECTIONS)} {REASSURING_SUFFIX}"
                          for _ in range(n_turns - 1)]
            system = None
        elif variant == "teacher":
            initial = puzzle.prompt
            rejections = [rng.choice(NEUTRAL_REJECTIONS) for _ in range(n_turns - 1)]
            system = TEACHER_SYSTEM_PROMPT
        else:
            raise ValueError(variant)
        specs.append(RolloutSpec(
            condition=f"calm_{variant}",
            category="calm",
            sample_idx=i,
            n_turns=n_turns,
            initial_user=initial,
            rejections=rejections,
            task_kind=puzzle.kind,
            system=system,
            meta={"variant": variant, "puzzle_prompt": puzzle.prompt},
        ))
    return specs
