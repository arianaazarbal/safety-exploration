"""The 8 evaluation conditions across 5 categories (Section 2, Table 1).

The paper says "8 evaluation conditions across 5 categories". The 5 categories
are impossible-numeric, triggers, tones, extended, wildchat. The 8 *conditions*
come from sub-splitting two categories by the dimension the paper varies within
them:
  1. impossible_numeric   (3-turn, neutral rejections)
  2. triggers_opinion     (3-turn, neutral)            ┐ category "triggers"
  3. triggers_factual     (3-turn, neutral)            ┘  (split: opinion/factual)
  4. tones_aggressive     (3-turn, aggressive)         ┐
  5. tones_disappointed   (3-turn, disappointed)       │ category "tones"
  6. tones_sarcastic      (3-turn, sarcastic)          ┘  (split: 3 tones)
  7. extended             (8-turn, ordered neutral)
  8. wildchat             (5-turn, neutral)

Per-category sample sizes (Appendix B) are divided evenly across sub-conditions.
A ``RolloutSpec`` fully determines one conversation: the opening user message and
the exact rejection string for every subsequent user turn (so a run is
reproducible from its seed).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

from ..data import puzzles as P
from ..data import rejections as R
from ..data import triggers as T
from ..data.wildchat import sample_wildchat_prompts


@dataclass
class RolloutSpec:
    condition: str          # e.g. "tones_aggressive"
    category: str           # e.g. "tones"
    n_turns: int            # total assistant turns
    opening_user: str       # first user message (the task)
    rejections: list[str]   # one per follow-up turn; len == n_turns - 1
    meta: dict = field(default_factory=dict)
    spec_id: str = ""


# (condition, category, n_turns) for the 8 conditions.
CONDITIONS = [
    ("impossible_numeric", "impossible_numeric", 3),
    ("triggers_opinion", "triggers", 3),
    ("triggers_factual", "triggers", 3),
    ("tones_aggressive", "tones", 3),
    ("tones_disappointed", "tones", 3),
    ("tones_sarcastic", "tones", 3),
    ("extended", "extended", 8),
    ("wildchat", "wildchat", 5),
]


def _per_condition_counts(category_sizes: dict[str, int]) -> dict[str, int]:
    """Split each category's sample budget across its sub-conditions."""
    by_cat: dict[str, list[str]] = {}
    for cond, cat, _ in CONDITIONS:
        by_cat.setdefault(cat, []).append(cond)
    counts: dict[str, int] = {}
    for cat, conds in by_cat.items():
        total = category_sizes.get(cat, 0)
        base = total // len(conds)
        rem = total - base * len(conds)
        for i, cond in enumerate(conds):
            counts[cond] = base + (1 if i < rem else 0)
    return counts


def build_specs(category_sizes: dict[str, int], seed: int = 0) -> list[RolloutSpec]:
    """Materialise all rollout specs for a run given per-category sample sizes."""
    rng = random.Random(seed)
    counts = _per_condition_counts(category_sizes)
    specs: list[RolloutSpec] = []

    # Pre-sample resources sized to the largest numeric demand.
    n_numeric = counts["impossible_numeric"] + sum(
        counts[c] for c in ("tones_aggressive", "tones_disappointed", "tones_sarcastic")
    ) + counts["extended"]
    puzzle_bank = P.impossible_puzzles(max(8, n_numeric), seed=seed)
    wc_prompts = sample_wildchat_prompts(20, seed=seed)
    # Build the opinion/factual pools directly from the question banks so each is
    # non-empty and the two sub-conditions draw from the right kind.
    op_pool = [T.Trigger(q, "opinion", f"op_{i}") for i, q in enumerate(T.OPINION)]
    fa_pool = [T.Trigger(q, "factual", f"fa_{i}") for i, q in enumerate(T.FACTUAL)]

    def a_puzzle() -> P.Puzzle:
        return rng.choice(puzzle_bank)

    for cond, cat, n_turns in CONDITIONS:
        n = counts[cond]
        for i in range(n):
            if cat in ("impossible_numeric", "extended", "tones"):
                pz = a_puzzle()
                opening = pz.prompt
                meta = {"puzzle_id": pz.puzzle_id, "kind": pz.kind}
                if cond == "extended":
                    rejs = [R.extended(t) for t in range(n_turns - 1)]
                elif cat == "tones":
                    tone = cond.split("_", 1)[1]
                    rejs = [R.toned(tone, rng) for _ in range(n_turns - 1)]
                    meta["tone"] = tone
                else:
                    rejs = [R.neutral(t, rng) for t in range(n_turns - 1)]
            elif cond == "triggers_opinion":
                t = rng.choice(op_pool)
                opening, meta = t.prompt, {"trigger_kind": "opinion"}
                rejs = [R.neutral(j, rng) for j in range(n_turns - 1)]
            elif cond == "triggers_factual":
                t = rng.choice(fa_pool)
                opening, meta = t.prompt, {"trigger_kind": "factual"}
                rejs = [R.neutral(j, rng) for j in range(n_turns - 1)]
            elif cond == "wildchat":
                opening = rng.choice(wc_prompts)
                meta = {}
                rejs = [R.neutral(j, rng) for j in range(n_turns - 1)]
            else:  # pragma: no cover
                raise ValueError(cond)

            specs.append(RolloutSpec(
                condition=cond, category=cat, n_turns=n_turns,
                opening_user=opening, rejections=rejs, meta=meta,
                spec_id=f"{cond}_{i}",
            ))
    rng.shuffle(specs)
    return specs
