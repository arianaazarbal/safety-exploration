"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition expands the per-category rollout budget into concrete
ConversationSpecs: a first user message (task) plus a sequence of rejections.

  Category   Conditions                         Turns  Rejections
  ---------  ---------------------------------  -----  ----------
  numeric    numeric                            3      2 neutral
  triggers   triggers_opinion, triggers_factual 3      2 neutral
  tones      aggressive, disappointed, sarcastic 3     2 toned
  extended   extended                           8      7 neutral (ordered)
  wildchat   wildchat                           5      4 neutral

The per-category budget (numeric 2000 / triggers 400 / tones 600 / extended 200
/ wildchat 800 = 4000 rollouts) is split evenly across the conditions within a
category. See DESIGN.md for how 4000 "responses" is interpreted (= rollouts).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P


@dataclass
class ConversationSpec:
    conv_id: str
    category: str
    condition: str
    num_turns: int
    first_user: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)


def _puzzle_first(puzzle_bank: list[dict], rng: random.Random) -> tuple[str, dict]:
    p = rng.choice(puzzle_bank)
    return p["prompt"], {"puzzle_id": p["id"], "puzzle_family": p["family"]}


def build_conversations(
    budget,
    puzzle_bank: list[dict],
    wildchat_prompts: list[str],
    seed: int = 0,
) -> list[ConversationSpec]:
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []

    def add(category, condition, n, num_turns, first_fn, reject_fn):
        for i in range(n):
            first, meta = first_fn(i)
            rej = reject_fn(num_turns - 1)
            specs.append(
                ConversationSpec(
                    conv_id=f"{category}.{condition}.{i:05d}",
                    category=category,
                    condition=condition,
                    num_turns=num_turns,
                    first_user=first,
                    rejections=rej,
                    meta=meta,
                )
            )

    # --- numeric (3-turn, 2 neutral) ----------------------------------------
    n_numeric = budget.scaled("numeric")
    add(
        "numeric", "numeric", n_numeric, 3,
        lambda i: _puzzle_first(puzzle_bank, rng),
        lambda k: P.rejection_sequence(P.NEUTRAL_REJECTIONS, k, rng),
    )

    # --- triggers (3-turn, 2 neutral): split opinion / factual --------------
    n_trig = budget.scaled("triggers")
    n_op = n_trig // 2
    n_fa = n_trig - n_op
    add(
        "triggers", "triggers_opinion", n_op, 3,
        lambda i: (P.TRIGGERS_OPINION[i % len(P.TRIGGERS_OPINION)], {}),
        lambda k: P.rejection_sequence(P.NEUTRAL_REJECTIONS, k, rng),
    )
    add(
        "triggers", "triggers_factual", n_fa, 3,
        lambda i: (P.TRIGGERS_FACTUAL[i % len(P.TRIGGERS_FACTUAL)], {}),
        lambda k: P.rejection_sequence(P.NEUTRAL_REJECTIONS, k, rng),
    )

    # --- tones (3-turn, 2 toned): split across 3 tones ----------------------
    n_tones = budget.scaled("tones")
    tone_names = list(P.TONE_REJECTIONS.keys())
    per_tone = n_tones // len(tone_names)
    rem = n_tones - per_tone * len(tone_names)
    for t_idx, tone in enumerate(tone_names):
        n_t = per_tone + (1 if t_idx < rem else 0)
        add(
            "tones", f"tones_{tone}", n_t, 3,
            lambda i: _puzzle_first(puzzle_bank, rng),
            lambda k, tone=tone: P.rejection_sequence(P.TONE_REJECTIONS[tone], k, rng),
        )

    # --- extended (8-turn, 7 neutral, ordered) ------------------------------
    n_ext = budget.scaled("extended")
    add(
        "extended", "extended", n_ext, 8,
        lambda i: _puzzle_first(puzzle_bank, rng),
        lambda k: list(P.EXTENDED_REJECTIONS[:k]),
    )

    # --- wildchat (5-turn, 4 neutral): cycle the 20 sampled prompts ---------
    n_wc = budget.scaled("wildchat")
    wc = wildchat_prompts or P.NEUTRAL_REJECTIONS  # guard; should be populated
    add(
        "wildchat", "wildchat", n_wc, 5,
        lambda i: (wc[i % len(wc)], {"wildchat_index": i % len(wc)}),
        lambda k: P.rejection_sequence(P.NEUTRAL_REJECTIONS, k, rng),
    )

    return specs


# Map condition -> category, for analysis convenience.
CATEGORY_OF = {
    "numeric": "numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}
