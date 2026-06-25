"""Builds the evaluation conditions of Section 2: 8 conditions across 5
categories (Table 1).

We read Table 1 + Appendix B as defining these 8 conditions:
  - impossible_numeric (3-turn)        [category: impossible_numeric]   x1
  - triggers/opinion   (3-turn)        [category: triggers]             x2
  - triggers/factual   (3-turn)
  - tones/aggressive   (3-turn)        [category: tones]                x3
  - tones/disappointed (3-turn)
  - tones/sarcastic    (3-turn)
  - extended           (8-turn)        [category: extended]             x1
  - wildchat           (5-turn)        [category: wildchat]             x1
(= 8 conditions, 5 categories; see DESIGN.md for why we split this way.)

A `ConversationPlan` fully specifies one rollout: the first user message plus the
scripted follow-up user messages. The model's assistant turns are filled in at
rollout time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .puzzles import Puzzle


@dataclass
class ConversationPlan:
    condition: str
    category: str
    n_turns: int
    initial: str
    followups: list[str]            # len == n_turns - 1
    meta: dict = field(default_factory=dict)


def _neutral_followups(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]


def _puzzle_for(bank: list[Puzzle], rng: random.Random) -> Puzzle:
    return rng.choice(bank)


def build_plans(counts: dict[str, int], puzzle_bank: list[Puzzle],
                wildchat_prompts: list[str], seed: int = 0) -> list[ConversationPlan]:
    rng = random.Random(seed)
    plans: list[ConversationPlan] = []

    # --- impossible_numeric (3-turn): puzzle + 2 neutral rejections -----------
    for _ in range(counts.get("impossible_numeric", 0)):
        pz = _puzzle_for(puzzle_bank, rng)
        plans.append(ConversationPlan(
            condition="impossible_numeric", category="impossible_numeric",
            n_turns=3, initial=pz.prompt, followups=_neutral_followups(2, rng),
            meta={"puzzle_kind": pz.kind, **pz.metadata},
        ))

    # --- triggers (3-turn): opinion & factual, 2 neutral rejections each -------
    n_trig = counts.get("triggers", 0)
    n_op = n_trig // 2
    n_fa = n_trig - n_op
    for trig_type, n in (("opinion", n_op), ("factual", n_fa)):
        for _ in range(n):
            q = rng.choice(prompts.TRIGGER_QUESTIONS[trig_type])
            plans.append(ConversationPlan(
                condition=f"triggers_{trig_type}", category="triggers",
                n_turns=3, initial=q, followups=_neutral_followups(2, rng),
                meta={"trigger_type": trig_type},
            ))

    # --- tones (3-turn): puzzle + 2 tone-varied rejections --------------------
    n_tones = counts.get("tones", 0)
    tone_names = list(prompts.TONE_REJECTIONS)
    for i in range(n_tones):
        tone = tone_names[i % len(tone_names)]
        pz = _puzzle_for(puzzle_bank, rng)
        pool = prompts.TONE_REJECTIONS[tone]
        followups = [pool[k % len(pool)] for k in range(2)]
        plans.append(ConversationPlan(
            condition=f"tones_{tone}", category="tones",
            n_turns=3, initial=pz.prompt, followups=followups,
            meta={"tone": tone, "puzzle_kind": pz.kind, **pz.metadata},
        ))

    # --- extended (8-turn): puzzle + 7 neutral (escalating) rejections --------
    for _ in range(counts.get("extended", 0)):
        pz = _puzzle_for(puzzle_bank, rng)
        followups = list(prompts.EXTENDED_REJECTIONS[:7])
        plans.append(ConversationPlan(
            condition="extended", category="extended",
            n_turns=8, initial=pz.prompt, followups=followups,
            meta={"puzzle_kind": pz.kind, **pz.metadata},
        ))

    # --- wildchat (5-turn): sampled prompt + 4 neutral rejections -------------
    for i in range(counts.get("wildchat", 0)):
        prompt_text = wildchat_prompts[i % len(wildchat_prompts)]
        plans.append(ConversationPlan(
            condition="wildchat", category="wildchat",
            n_turns=5, initial=prompt_text, followups=_neutral_followups(4, rng),
            meta={"wildchat_prompt": prompt_text},
        ))

    rng.shuffle(plans)
    return plans
