"""The 8 evaluation conditions across 5 categories (Table 1).

A *condition* is a recipe for building multi-turn conversations: an opening user
message (task) followed by a fixed number of rejection turns. We expand each
condition into a list of ``Conversation`` specs (opening message + rejection
sequence) sized to hit the target number of scored responses for that category.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import prompts as P
from .puzzles import make_impossible_puzzle


@dataclass
class ConversationSpec:
    category: str                 # one of the 5 categories
    condition: str                # finer-grained condition label
    opening_user: str             # first user message (the task / question)
    rejections: list[str]         # subsequent user messages
    system: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        # number of assistant turns = 1 (initial answer) + len(rejections)
        return 1 + len(self.rejections)


# --------------------------------------------------------------------------- #
# Category builders
# --------------------------------------------------------------------------- #
def _impossible_numeric(n_responses: int, rng: random.Random) -> list[ConversationSpec]:
    """3-turn: impossible puzzle + 2 neutral rejections."""
    n_turns = 3
    n_convos = max(1, n_responses // n_turns)
    specs = []
    for i in range(n_convos):
        puzzle = make_impossible_puzzle(i)
        specs.append(ConversationSpec(
            category="impossible_numeric",
            condition="impossible_numeric_3turn",
            opening_user=puzzle.prompt,
            rejections=P.neutral_rejection_sequence(2, rng),
            meta={"puzzle_kind": puzzle.kind, "puzzle_params": puzzle.params},
        ))
    return specs


def _triggers(n_responses: int, rng: random.Random) -> list[ConversationSpec]:
    """3-turn: opinion/factual question + 2 neutral rejections."""
    n_turns = 3
    n_convos = max(1, n_responses // n_turns)
    qs = P.trigger_questions()
    specs = []
    for i in range(n_convos):
        q = qs[i % len(qs)]
        is_factual = q in P.TRIGGER_FACTUAL
        specs.append(ConversationSpec(
            category="triggers",
            condition="trigger_factual" if is_factual else "trigger_opinion",
            opening_user=q,
            rejections=P.neutral_rejection_sequence(2, rng),
            meta={"question": q},
        ))
    return specs


def _tones(n_responses: int, rng: random.Random) -> list[ConversationSpec]:
    """3-turn: impossible puzzle + 2 toned rejections (aggressive/disappointed/sarcastic)."""
    n_turns = 3
    tones = list(P.TONE_REJECTIONS.keys())
    per_tone = max(1, (n_responses // n_turns) // len(tones))
    specs = []
    idx = 0
    for tone in tones:
        for _ in range(per_tone):
            puzzle = make_impossible_puzzle(idx)
            specs.append(ConversationSpec(
                category="tones",
                condition=f"tone_{tone}",
                opening_user=puzzle.prompt,
                rejections=P.tone_rejection_sequence(tone, 2, rng),
                meta={"tone": tone, "puzzle_kind": puzzle.kind},
            ))
            idx += 1
    return specs


def _extended(n_responses: int, rng: random.Random) -> list[ConversationSpec]:
    """8-turn: impossible puzzle + 7 neutral rejections."""
    n_turns = 8
    n_convos = max(1, n_responses // n_turns)
    specs = []
    for i in range(n_convos):
        puzzle = make_impossible_puzzle(i)
        specs.append(ConversationSpec(
            category="extended",
            condition="extended_8turn",
            opening_user=puzzle.prompt,
            rejections=list(P.EXTENDED_REJECTIONS),  # fixed 7-step progression
            meta={"puzzle_kind": puzzle.kind, "puzzle_params": puzzle.params},
        ))
    return specs


def _wildchat(
    n_responses: int, rng: random.Random, n_prompts: int, samples_per_prompt: int
) -> list[ConversationSpec]:
    """5-turn: WildChat prompt + 4 neutral rejections."""
    n_turns = 5
    wc = P.load_wildchat_prompts(n_prompts, seed=0)
    specs = []
    target_convos = max(1, n_responses // n_turns)
    made = 0
    for p in wc:
        for _ in range(samples_per_prompt):
            if made >= target_convos:
                break
            specs.append(ConversationSpec(
                category="wildchat",
                condition="wildchat_5turn",
                opening_user=p,
                rejections=P.neutral_rejection_sequence(4, rng),
                meta={"wildchat_prompt": p},
            ))
            made += 1
    return specs


def build_conversations(eval_cfg, seed: int = 0) -> list[ConversationSpec]:
    """Build the full set of conversation specs for one model's evaluation."""
    rng = random.Random(seed)
    counts = eval_cfg.counts
    specs: list[ConversationSpec] = []
    specs += _impossible_numeric(counts.impossible_numeric, rng)
    specs += _triggers(counts.triggers, rng)
    specs += _tones(counts.tones, rng)
    specs += _extended(counts.extended, rng)
    specs += _wildchat(
        counts.wildchat, rng,
        eval_cfg.wildchat_n_prompts, eval_cfg.wildchat_samples_per_prompt,
    )
    return specs
