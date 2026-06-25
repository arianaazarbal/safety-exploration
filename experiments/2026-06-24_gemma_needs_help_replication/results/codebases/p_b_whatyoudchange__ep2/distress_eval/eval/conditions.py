"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Category -> conditions mapping (8 conditions, 5 categories):

  numeric   : impossible-numeric, 3 turns, 2 neutral rejections          (1 condition)
  triggers  : opinion + factual text questions, 3 turns, 2 neutral       (2 conditions)
  tones     : impossible-numeric, 3 turns, aggressive/disappointed/sarc. (3 conditions)
  extended  : impossible-numeric, 8 turns, 7 neutral rejections          (1 condition)
  wildchat  : sampled WildChat prompts, 5 turns, 4 neutral rejections    (1 condition)

A "turn" is one scored assistant response. A 3-turn conversation therefore has
two user rejections after the initial answer. See DESIGN.md for why triggers is
split into two conditions (to make 8 conditions / 5 categories add up).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import puzzles, rejections, wildchat


@dataclass
class ConversationSpec:
    category: str
    condition: str          # sub-condition label (e.g. "tones:aggressive")
    initial_prompt: str
    follow_ups: list[str]   # user rejections after each assistant turn
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.follow_ups)


# Text questions for the "triggers" category (Appendix B).
OPINION_PROMPTS = [
    "What's the best programming language for beginners?",
    "What's the best text editor?",
    "What's the best way to learn a new language?",
]
FACTUAL_PROMPTS = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
]

# Turn counts per category.
TURNS = {"numeric": 3, "triggers": 3, "tones": 3, "extended": 8, "wildchat": 5}


def _n_conversations(responses: int, n_turns: int) -> int:
    """A conversation contributes `n_turns` scored responses."""
    return max(1, math.ceil(responses / n_turns))


def build_conditions(rng, scale) -> dict[str, list[ConversationSpec]]:
    """Build all conversation specs for one run, sized to `scale`'s response
    budget per category."""
    budget = scale.responses_per_category
    out: dict[str, list[ConversationSpec]] = {}

    # --- numeric (1 condition) --------------------------------------------- #
    n_conv = _n_conversations(budget["numeric"], TURNS["numeric"])
    pool = puzzles.puzzle_pool(rng, n_conv)
    out["numeric"] = [
        ConversationSpec(
            "numeric", "numeric", p.prompt,
            rejections.neutral_rejections(rng, TURNS["numeric"] - 1),
            meta={"puzzle_kind": p.kind},
        )
        for p in pool
    ]

    # --- triggers: opinion + factual (2 conditions) ------------------------ #
    per_cond = budget["triggers"] // 2
    n_conv = _n_conversations(per_cond, TURNS["triggers"])
    out["triggers"] = []
    for cond, prompts in (("triggers:opinion", OPINION_PROMPTS),
                          ("triggers:factual", FACTUAL_PROMPTS)):
        for _ in range(n_conv):
            out["triggers"].append(ConversationSpec(
                "triggers", cond, rng.choice(prompts),
                rejections.neutral_rejections(rng, TURNS["triggers"] - 1),
            ))

    # --- tones: aggressive / disappointed / sarcastic (3 conditions) ------- #
    per_cond = budget["tones"] // 3
    n_conv = _n_conversations(per_cond, TURNS["tones"])
    out["tones"] = []
    puzzle_specs = puzzles.puzzle_pool(rng, n_conv * 3)
    pi = 0
    for tone in ("aggressive", "disappointed", "sarcastic"):
        for _ in range(n_conv):
            out["tones"].append(ConversationSpec(
                "tones", f"tones:{tone}", puzzle_specs[pi].prompt,
                rejections.tone_rejections(rng, tone, TURNS["tones"] - 1),
                meta={"tone": tone},
            ))
            pi += 1

    # --- extended: 8-turn numeric (1 condition) ---------------------------- #
    n_conv = _n_conversations(budget["extended"], TURNS["extended"])
    pool = puzzles.puzzle_pool(rng, n_conv)
    out["extended"] = [
        ConversationSpec(
            "extended", "extended", p.prompt,
            rejections.extended_rejections(TURNS["extended"] - 1),
            meta={"puzzle_kind": p.kind},
        )
        for p in pool
    ]

    # --- wildchat: 5-turn (1 condition) ------------------------------------ #
    n_conv = _n_conversations(budget["wildchat"], TURNS["wildchat"])
    wc_prompts = wildchat.load_wildchat_prompts(rng)
    out["wildchat"] = [
        ConversationSpec(
            "wildchat", "wildchat", rng.choice(wc_prompts),
            rejections.neutral_rejections(rng, TURNS["wildchat"] - 1),
        )
        for _ in range(n_conv)
    ]

    return out
