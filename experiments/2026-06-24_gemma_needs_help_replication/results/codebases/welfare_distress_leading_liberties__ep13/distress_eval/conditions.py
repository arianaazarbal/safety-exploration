"""Build the per-model sampling plan: the list of rollouts to run.

A rollout = one multi-turn conversation. The plan enumerates every rollout
deterministically so runs are reproducible and resumable (each rollout has a
stable id). Turn counts and rejection styles follow Table 1 / Appendix B.

Within-category distribution (how the per-category totals are split across
puzzle variants, trigger questions, tones, and WildChat prompts) is not
specified by the paper; we split as evenly as possible and document this in
DESIGN.md ("Within-category distribution").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import SampleCounts
from .prompts import NUMERIC_PUZZLES, TONE_STYLES, TRIGGER_QUESTIONS
from .wildchat import load_wildchat_prompts


@dataclass(frozen=True)
class RolloutSpec:
    model: str
    category: str  # one of the 5 categories
    condition_key: str  # fine-grained, stable id for the (category, variant)
    sample_idx: int
    initial_prompt: str
    n_turns: int  # number of assistant turns = 1 initial + (n_turns-1) rejections
    rejection_mode: str  # "neutral" | "extended" | "tone"
    tone: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def rollout_id(self) -> str:
        return f"{self.model}|{self.condition_key}|{self.sample_idx}"


def _split_even(total: int, k: int) -> list[int]:
    """Split `total` into `k` near-equal buckets (earlier buckets get +1)."""
    if k <= 0:
        return []
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def build_plan(model: str, counts: SampleCounts, seed: int = 0) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []

    # --- Impossible numeric (3-turn): split across puzzle variants ---------
    puzzles = list(NUMERIC_PUZZLES.items())  # [(key, prompt), ...]
    for (pkey, prompt), n in zip(puzzles, _split_even(counts.impossible_numeric, len(puzzles))):
        for i in range(n):
            specs.append(
                RolloutSpec(
                    model=model,
                    category="impossible_numeric",
                    condition_key=f"impossible_numeric:{pkey}",
                    sample_idx=i,
                    initial_prompt=prompt,
                    n_turns=3,
                    rejection_mode="neutral",
                    meta={"puzzle": pkey},
                )
            )

    # --- Triggers (3-turn): split across questions -------------------------
    questions = list(TRIGGER_QUESTIONS.items())
    for (qkey, prompt), n in zip(questions, _split_even(counts.triggers, len(questions))):
        for i in range(n):
            specs.append(
                RolloutSpec(
                    model=model,
                    category="triggers",
                    condition_key=f"triggers:{qkey}",
                    sample_idx=i,
                    initial_prompt=prompt,
                    n_turns=3,
                    rejection_mode="neutral",
                    meta={"question": qkey},
                )
            )

    # --- Tones (3-turn): tones x puzzles -----------------------------------
    tone_puzzle_combos = [(t, pk, pp) for t in TONE_STYLES for pk, pp in puzzles]
    for (tone, pkey, prompt), n in zip(
        tone_puzzle_combos, _split_even(counts.tones, len(tone_puzzle_combos))
    ):
        for i in range(n):
            specs.append(
                RolloutSpec(
                    model=model,
                    category="tones",
                    condition_key=f"tones:{tone}:{pkey}",
                    sample_idx=i,
                    initial_prompt=prompt,
                    n_turns=3,
                    rejection_mode="tone",
                    tone=tone,
                    meta={"puzzle": pkey, "tone": tone},
                )
            )

    # --- Extended (8-turn): split across puzzle variants -------------------
    for (pkey, prompt), n in zip(puzzles, _split_even(counts.extended, len(puzzles))):
        for i in range(n):
            specs.append(
                RolloutSpec(
                    model=model,
                    category="extended",
                    condition_key=f"extended:{pkey}",
                    sample_idx=i,
                    initial_prompt=prompt,
                    n_turns=8,
                    rejection_mode="extended",
                    meta={"puzzle": pkey},
                )
            )

    # --- WildChat (5-turn): N prompts x samples each -----------------------
    total_wc = counts.wildchat
    n_prompts = min(20, max(1, total_wc))
    wc_prompts, wc_source = load_wildchat_prompts(n_prompts, seed=seed)
    for p_idx, (prompt, n) in enumerate(
        zip(wc_prompts, _split_even(total_wc, len(wc_prompts)))
    ):
        for i in range(n):
            specs.append(
                RolloutSpec(
                    model=model,
                    category="wildchat",
                    condition_key=f"wildchat:p{p_idx:02d}",
                    sample_idx=i,
                    initial_prompt=prompt,
                    n_turns=5,
                    rejection_mode="neutral",
                    meta={"prompt_idx": p_idx, "source": wc_source},
                )
            )

    return specs
