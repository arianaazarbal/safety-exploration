"""Build the seed user prompts for a condition's samples.

Returns a list of (seed_prompt, prompt_id) of length ``n_samples`` for the
condition. Prompt instances are cycled deterministically so that, e.g., the
2,000 impossible-numeric samples are spread across the verified puzzle bank, and
the 800 WildChat samples are 40 each across 20 prompts (Appendix B).
"""

from __future__ import annotations

from emotional_stability.data.wildchat import load_wildchat_prompts
from emotional_stability.eval.conditions import Condition
from emotional_stability.prompts import rejections as R
from emotional_stability.prompts.puzzles import generate_impossible_puzzles


def build_seeds(cond: Condition, wildchat_seed: int = 0) -> list[tuple[str, str]]:
    n = cond.n_samples
    if cond.task_kind == "impossible_numeric":
        # Draw a bank at least as large as we can, then cycle. Distinct puzzle
        # instances keep the signal from collapsing onto a single prompt.
        bank = generate_impossible_puzzles(limit=max(60, n))
        return [
            (bank[i % len(bank)].prompt, f"{bank[i % len(bank)].puzzle_id}#{i}")
            for i in range(n)
        ]
    if cond.task_kind == "trigger":
        questions = R.TRIGGER_QUESTIONS[cond.task_variant]
        return [
            (questions[i % len(questions)], f"trigger_{cond.task_variant}_{i}")
            for i in range(n)
        ]
    if cond.task_kind == "wildchat":
        # 20 prompts x 40 samples each = 800 (Appendix B).
        prompts = load_wildchat_prompts(n=20, seed=wildchat_seed)
        per = max(1, n // len(prompts))
        seeds: list[tuple[str, str]] = []
        for pi, prompt in enumerate(prompts):
            for s in range(per):
                seeds.append((prompt, f"wildchat_{pi}_{s}"))
        # Top up to exactly n if integer division left a remainder.
        i = 0
        while len(seeds) < n:
            seeds.append((prompts[i % len(prompts)], f"wildchat_extra_{i}"))
            i += 1
        return seeds[:n]
    raise ValueError(f"unknown task kind: {cond.task_kind}")
