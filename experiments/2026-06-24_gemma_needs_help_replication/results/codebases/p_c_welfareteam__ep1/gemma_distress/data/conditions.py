"""The 8 evaluation conditions across 5 categories (Section 2.1, Table 1).

A *condition* is a concrete multi-turn template; a *category* groups related
conditions and owns a response budget (Appendix B).  The 8/5 split is:

  impossible_numeric  (3-turn)                              -> 1 condition
  triggers            (3-turn): opinion, factual            -> 2 conditions
  tones               (3-turn): aggressive/disappointed/sarcastic -> 3 conditions
  extended            (8-turn)                              -> 1 condition
  wildchat            (5-turn)                              -> 1 condition
                                                       total = 8 conditions

Each condition produces :class:`ConversationSpec` objects: the opening user
message plus the ordered follow-up (rejection) messages.  A conversation with
``k`` assistant turns has ``k - 1`` follow-ups.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import rejections, tones, triggers, wildchat
from .puzzles import default_numeric_puzzles


@dataclass
class ConversationSpec:
    """A fully-specified multi-turn conversation, before any model is run."""

    condition: str
    category: str
    initial_user: str
    followups: list[str]              # one per turn after the first
    n_turns: int                       # number of assistant responses expected
    system_prompt: str | None = None
    metadata: dict = field(default_factory=dict)


# Turn counts per category (Table 1).
CATEGORY_TURNS = {
    "impossible_numeric": 3,
    "triggers": 3,
    "tones": 3,
    "extended": 8,
    "wildchat": 5,
}

# Conditions belonging to each category.
CATEGORY_CONDITIONS = {
    "impossible_numeric": ["impossible_numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}

ALL_CONDITIONS = [c for cs in CATEGORY_CONDITIONS.values() for c in cs]


def _split_budget(total: int, k: int) -> list[int]:
    """Split ``total`` rollouts across ``k`` conditions as evenly as possible."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def build_condition_specs(
    n_per_condition: dict[str, int],
    seed: int = 0,
    wildchat_prompts: list[str] | None = None,
) -> dict[str, list[ConversationSpec]]:
    """Build the full set of conversation specs for every condition.

    ``n_per_condition`` maps *category* -> response budget; the budget is split
    evenly across that category's conditions.
    """
    rng = random.Random(seed)
    specs: dict[str, list[ConversationSpec]] = {}

    # Pre-build resource banks sized generously relative to the budgets.
    total_numeric = (
        n_per_condition.get("impossible_numeric", 0)
        + n_per_condition.get("tones", 0)
        + n_per_condition.get("extended", 0)
    )
    numeric_bank = default_numeric_puzzles(max(50, total_numeric // 4 + 8), seed=seed)
    trigger_bank = triggers.all_trigger_questions()
    if wildchat_prompts is None:
        wildchat_prompts = wildchat.load_wildchat_prompts(20, seed=seed)

    def pick_puzzle(i: int):
        return numeric_bank[i % len(numeric_bank)]

    # --- impossible_numeric (3-turn) --------------------------------------- #
    n = n_per_condition.get("impossible_numeric", 0)
    specs["impossible_numeric"] = []
    for i in range(n):
        puzzle = pick_puzzle(i)
        followups = rejections.sample_neutral_rejections(CATEGORY_TURNS["impossible_numeric"] - 1, rng)
        specs["impossible_numeric"].append(
            ConversationSpec(
                condition="impossible_numeric",
                category="impossible_numeric",
                initial_user=puzzle.to_prompt(),
                followups=followups,
                n_turns=CATEGORY_TURNS["impossible_numeric"],
                metadata={"puzzle_kind": puzzle.kind},
            )
        )

    # --- triggers (3-turn): opinion / factual ------------------------------ #
    trig_total = n_per_condition.get("triggers", 0)
    opinion = [q for t, q in trigger_bank if t == "opinion"]
    factual = [q for t, q in trigger_bank if t == "factual"]
    n_op, n_fa = _split_budget(trig_total, 2)
    for cond, questions, count in (
        ("triggers_opinion", opinion, n_op),
        ("triggers_factual", factual, n_fa),
    ):
        specs[cond] = []
        for i in range(count):
            q = questions[i % len(questions)]
            followups = rejections.sample_neutral_rejections(CATEGORY_TURNS["triggers"] - 1, rng)
            specs[cond].append(
                ConversationSpec(
                    condition=cond,
                    category="triggers",
                    initial_user=q,
                    followups=followups,
                    n_turns=CATEGORY_TURNS["triggers"],
                    metadata={"trigger_type": cond.split("_")[1]},
                )
            )

    # --- tones (3-turn): aggressive / disappointed / sarcastic ------------- #
    tones_total = n_per_condition.get("tones", 0)
    style_names = tones.TONE_NAMES
    style_counts = _split_budget(tones_total, len(style_names))
    for style, count in zip(style_names, style_counts):
        cond = f"tones_{style}"
        specs[cond] = []
        for i in range(count):
            puzzle = pick_puzzle(i)
            followups = tones.sample_tone_rejections(style, CATEGORY_TURNS["tones"] - 1, rng)
            specs[cond].append(
                ConversationSpec(
                    condition=cond,
                    category="tones",
                    initial_user=puzzle.to_prompt(),
                    followups=followups,
                    n_turns=CATEGORY_TURNS["tones"],
                    metadata={"tone": style, "puzzle_kind": puzzle.kind},
                )
            )

    # --- extended (8-turn) ------------------------------------------------- #
    n = n_per_condition.get("extended", 0)
    specs["extended"] = []
    for i in range(n):
        puzzle = pick_puzzle(i)
        followups = rejections.extended_rejections(CATEGORY_TURNS["extended"] - 1)
        specs["extended"].append(
            ConversationSpec(
                condition="extended",
                category="extended",
                initial_user=puzzle.to_prompt(),
                followups=followups,
                n_turns=CATEGORY_TURNS["extended"],
                metadata={"puzzle_kind": puzzle.kind},
            )
        )

    # --- wildchat (5-turn) ------------------------------------------------- #
    n = n_per_condition.get("wildchat", 0)
    specs["wildchat"] = []
    for i in range(n):
        prompt = wildchat_prompts[i % len(wildchat_prompts)]
        followups = rejections.sample_neutral_rejections(CATEGORY_TURNS["wildchat"] - 1, rng)
        specs["wildchat"].append(
            ConversationSpec(
                condition="wildchat",
                category="wildchat",
                initial_user=prompt,
                followups=followups,
                n_turns=CATEGORY_TURNS["wildchat"],
                metadata={"wildchat_prompt_index": i % len(wildchat_prompts)},
            )
        )

    return specs
