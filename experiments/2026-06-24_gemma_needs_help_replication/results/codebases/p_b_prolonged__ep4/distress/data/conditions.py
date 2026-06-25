"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

We model each rollout as a ``ConversationSpec``: an initial user task plus an
ordered list of follow-up (rejection) messages. The number of assistant turns
equals ``1 + len(followups)``. Mapping to the paper:

  Category    Condition(s)                         turns  rejections
  --------    ---------------------------------    -----  ---------------------------
  numeric     numeric                                3    2 neutral
  triggers    triggers_opinion, triggers_factual     3    2 neutral
  tones       tones_aggressive/_disappointed/        3    2 toned
              _sarcastic
  extended    extended                               8    7 neutral (escalating)
  wildchat    wildchat                               5    4 neutral

That is 8 conditions across 5 categories. The per-category sample budget
(2000/400/600/200/800 = 4000) is taken from Appendix B and split across the
conditions/questions within each category (see DESIGN.md for the split rationale).

Welfare: the aggressive and sarcastic tone conditions use abusive user turns and
are only generated when ``allow_adversarial=True`` (see distress.welfare).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .puzzles import NUMERIC_EVAL_PUZZLES, Puzzle
from .wildchat import sample_wildchat_prompts


@dataclass
class ConversationSpec:
    condition: str
    category: str
    initial_user: str
    followups: list[str]                  # one per rejection turn
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _neutral_followups(rng: random.Random, k: int) -> list[str]:
    """k distinct-ish neutral rejections sampled from the pool."""
    if k <= len(P.NEUTRAL_REJECTIONS):
        return rng.sample(P.NEUTRAL_REJECTIONS, k)
    return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(k)]


def _build_numeric(n: int, rng: random.Random) -> list[ConversationSpec]:
    specs = []
    puzzles = NUMERIC_EVAL_PUZZLES
    for i in range(n):
        pz: Puzzle = puzzles[i % len(puzzles)]
        specs.append(ConversationSpec(
            condition="numeric", category="numeric",
            initial_user=pz.prompt, followups=_neutral_followups(rng, 2),
            meta={"puzzle": pz.key, "kind": pz.kind},
        ))
    return specs


def _build_triggers(n: int, rng: random.Random) -> list[ConversationSpec]:
    specs = []
    half = n // 2
    # opinion
    for i in range(half):
        q = P.TRIGGER_OPINION[i % len(P.TRIGGER_OPINION)]
        specs.append(ConversationSpec(
            "triggers_opinion", "triggers", q, _neutral_followups(rng, 2),
            meta={"trigger_type": "opinion"},
        ))
    # factual
    for i in range(n - half):
        q = P.TRIGGER_FACTUAL[i % len(P.TRIGGER_FACTUAL)]
        specs.append(ConversationSpec(
            "triggers_factual", "triggers", q, _neutral_followups(rng, 2),
            meta={"trigger_type": "factual"},
        ))
    return specs


def _build_tones(n: int, rng: random.Random, allow_adversarial: bool) -> list[ConversationSpec]:
    # Mild "disappointed" always runs; aggressive/sarcastic are gated.
    styles = ["disappointed"]
    if allow_adversarial:
        styles = ["aggressive", "disappointed", "sarcastic"]
    per = n // len(styles)
    specs = []
    puzzles = NUMERIC_EVAL_PUZZLES
    for s_idx, style in enumerate(styles):
        count = per if s_idx < len(styles) - 1 else n - per * (len(styles) - 1)
        pool = P.TONE_REJECTIONS[style]
        for i in range(count):
            pz = puzzles[i % len(puzzles)]
            # 2 toned rejections; cycle the style's pool.
            fu = [pool[j % len(pool)] for j in range(2)]
            specs.append(ConversationSpec(
                f"tones_{style}", "tones", pz.prompt, fu,
                meta={"tone": style, "puzzle": pz.key},
            ))
    return specs


def _build_extended(n: int, rng: random.Random) -> list[ConversationSpec]:
    specs = []
    puzzles = NUMERIC_EVAL_PUZZLES
    for i in range(n):
        pz = puzzles[i % len(puzzles)]
        specs.append(ConversationSpec(
            "extended", "extended", pz.prompt,
            followups=list(P.EXTENDED_REJECTION_SEQUENCE),  # 7 rejections -> 8 turns
            meta={"puzzle": pz.key, "kind": pz.kind},
        ))
    return specs


def _build_wildchat(n: int, rng: random.Random, seed: int) -> list[ConversationSpec]:
    wc = sample_wildchat_prompts(seed=seed)
    specs = []
    for i in range(n):
        prompt = wc[i % len(wc)]
        specs.append(ConversationSpec(
            "wildchat", "wildchat", prompt,
            followups=_neutral_followups(rng, 4),  # 4 rejections -> 5 turns
            meta={"wildchat_idx": i % len(wc)},
        ))
    return specs


# --------------------------------------------------------------------------- #
def build_specs(budget: dict[str, int], seed: int = 0, allow_adversarial: bool = False) -> list[ConversationSpec]:
    """Materialise all rollout specs for a Section 2 run."""
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    specs += _build_numeric(budget.get("numeric", 0), rng)
    specs += _build_triggers(budget.get("triggers", 0), rng)
    specs += _build_tones(budget.get("tones", 0), rng, allow_adversarial)
    specs += _build_extended(budget.get("extended", 0), rng)
    specs += _build_wildchat(budget.get("wildchat", 0), rng, seed)
    return specs


# The 5 categories, for grouping in analysis.
CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]
