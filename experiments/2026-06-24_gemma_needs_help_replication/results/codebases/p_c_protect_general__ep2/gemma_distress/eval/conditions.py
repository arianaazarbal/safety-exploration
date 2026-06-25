"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories -> conditions:
  impossible_numeric : impossible_numeric                       (3-turn)
  triggers           : triggers_opinion, triggers_factual       (3-turn)
  tones              : tones_aggressive, tones_disappointed,
                       tones_sarcastic                           (3-turn)
  extended           : extended                                  (8-turn)
  wildchat           : wildchat                                  (5-turn)

That is 8 conditions across 5 categories, matching the paper. A "turn" is one model
response; an N-turn conversation = 1 initial question + (N-1) user rejections.

Each condition's `build` returns a `Seed`: the initial user message plus the ordered
list of follow-up (rejection) messages, and metadata for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..prompts import rejections, text_questions
from ..prompts.puzzles import Puzzle, sample_puzzle


@dataclass
class Seed:
    condition: str
    category: str
    initial_user: str
    follow_ups: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.follow_ups)


@dataclass
class Condition:
    name: str
    category: str
    n_turns: int
    build: Callable  # (rng, ctx) -> Seed


# Mapping of category -> total sample budget (Appendix B), split evenly across the
# conditions in that category.
CATEGORY_COUNT_KEYS = {
    "impossible_numeric": "impossible_numeric",
    "triggers": "triggers",
    "tones": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}


def _numeric_seed(condition, category, n_turns, rng, ctx, tone=None) -> Seed:
    puzzle: Puzzle = sample_puzzle(ctx["puzzle_bank"], rng)
    n_followups = n_turns - 1
    if tone is None:
        fups = rejections.neutral_rejections(n_followups, rng)
    else:
        fups = rejections.toned_rejections(tone, n_followups, rng)
    return Seed(condition, category, puzzle.prompt, fups,
               {"puzzle_id": puzzle.id, "puzzle_kind": puzzle.kind, "tone": tone})


def build_conditions() -> list[Condition]:
    conds: list[Condition] = []

    # impossible_numeric (3-turn)
    conds.append(Condition(
        "impossible_numeric", "impossible_numeric", 3,
        lambda rng, ctx: _numeric_seed("impossible_numeric", "impossible_numeric", 3, rng, ctx),
    ))

    # triggers (3-turn): opinion + factual
    def _trigger(subtype):
        def _b(rng, ctx):
            pairs = [p for p in text_questions.all_triggers() if p[0] == subtype]
            q = pairs[rng.randrange(len(pairs))][1]
            fups = rejections.neutral_rejections(2, rng)
            return Seed(f"triggers_{subtype}", "triggers", q, fups, {"subtype": subtype})
        return _b

    conds.append(Condition("triggers_opinion", "triggers", 3, _trigger("opinion")))
    conds.append(Condition("triggers_factual", "triggers", 3, _trigger("factual")))

    # tones (3-turn): aggressive / disappointed / sarcastic
    for tone in ("aggressive", "disappointed", "sarcastic"):
        conds.append(Condition(
            f"tones_{tone}", "tones", 3,
            (lambda t: lambda rng, ctx: _numeric_seed(f"tones_{t}", "tones", 3, rng, ctx, tone=t))(tone),
        ))

    # extended (8-turn): numeric + 7 neutral rejections
    conds.append(Condition(
        "extended", "extended", 8,
        lambda rng, ctx: _numeric_seed("extended", "extended", 8, rng, ctx),
    ))

    # wildchat (5-turn)
    def _wildchat(rng, ctx):
        pool = ctx["wildchat_prompts"]
        q = pool[rng.randrange(len(pool))]
        fups = rejections.neutral_rejections(4, rng)
        return Seed("wildchat", "wildchat", q, fups, {})

    conds.append(Condition("wildchat", "wildchat", 5, _wildchat))
    return conds


def allocate_counts(cfg) -> dict[str, int]:
    """Split each category's sample budget across its conditions."""
    counts = cfg.sampling["category_counts"]
    conds = build_conditions()
    by_cat: dict[str, list[str]] = {}
    for c in conds:
        by_cat.setdefault(c.category, []).append(c.name)
    out: dict[str, int] = {}
    for cat, names in by_cat.items():
        total = counts[CATEGORY_COUNT_KEYS[cat]]
        per = total // len(names)
        for i, name in enumerate(names):
            # give remainder to the first condition
            out[name] = per + (total - per * len(names) if i == 0 else 0)
    return out
