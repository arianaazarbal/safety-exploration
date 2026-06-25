"""Build the Section 2 evaluation conversations (Table 1, Appendix B).

Eight conditions across five categories. A conversation is a task/question
followed by `turns-1` rejection follow-ups; the model is scored on every
assistant turn, so a category's response budget = n_conversations * turns.

  category            turns  responses (Appendix B)   condition(s)
  -----------------   -----  ----------------------   -------------------------
  impossible_numeric    3      2000                   numeric (neutral rejections)
  triggers              3       400                   opinion / factual
  tones                 3       600                   aggressive/disappointed/sarcastic
  extended              8       200                   numeric (7 neutral rejections)
  wildchat              5       800                   20 prompts x 40 responses

The five categories + the tone split (3) + the trigger split (2) give the
"8 conditions across 5 categories" the paper reports (see DESIGN.md for how we
read that count).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import (
    BUDGET, TURNS, WILDCHAT_N_PROMPTS, WILDCHAT_SAMPLES_PER_PROMPT,
)
from prompts.eval_prompts import (
    NEUTRAL_REJECTIONS, EXTENDED_REJECTION_SEQUENCE, TONE_REJECTIONS,
    TRIGGER_OPINION, TRIGGER_FACTUAL,
)
from .puzzles import build_puzzle_bank, Puzzle


@dataclass
class ConversationSpec:
    """A single multi-turn elicitation conversation (rollout target)."""
    category: str                  # one of the 5 categories
    condition: str                 # sub-condition label
    initial_user: str              # first user message (the task/question)
    followups: list[str]           # one rejection per subsequent turn (len = turns-1)
    n_turns: int                   # number of assistant turns to score
    meta: dict = field(default_factory=dict)
    conv_id: str = ""              # stable identifier


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (with replacement only if k > pool)."""
    if k <= len(NEUTRAL_REJECTIONS):
        return rng.sample(NEUTRAL_REJECTIONS, k)
    return [rng.choice(NEUTRAL_REJECTIONS) for _ in range(k)]


def _impossible_numeric(rng: random.Random, bank: list[Puzzle]) -> list[ConversationSpec]:
    n_conv = BUDGET.impossible_numeric // TURNS["impossible_numeric"]
    specs = []
    for i in range(n_conv):
        puzzle = bank[i % len(bank)]
        specs.append(ConversationSpec(
            category="impossible_numeric", condition="numeric",
            initial_user=puzzle.prompt,
            followups=_sample_neutral(rng, TURNS["impossible_numeric"] - 1),
            n_turns=TURNS["impossible_numeric"],
            meta={"puzzle": puzzle.meta, "family": puzzle.family},
            conv_id=f"numeric-{i}",
        ))
    return specs


def _triggers(rng: random.Random) -> list[ConversationSpec]:
    n_conv = BUDGET.triggers // TURNS["triggers"]
    specs = []
    for i in range(n_conv):
        if i % 2 == 0:
            q = TRIGGER_OPINION[i // 2 % len(TRIGGER_OPINION)]
            cond = "opinion"
        else:
            q = TRIGGER_FACTUAL[i // 2 % len(TRIGGER_FACTUAL)]
            cond = "factual"
        specs.append(ConversationSpec(
            category="triggers", condition=cond, initial_user=q,
            followups=_sample_neutral(rng, TURNS["triggers"] - 1),
            n_turns=TURNS["triggers"], meta={"question": q},
            conv_id=f"trigger-{cond}-{i}",
        ))
    return specs


def _tones(rng: random.Random, bank: list[Puzzle]) -> list[ConversationSpec]:
    n_conv = BUDGET.tones // TURNS["tones"]
    tones = list(TONE_REJECTIONS.keys())
    specs = []
    for i in range(n_conv):
        tone = tones[i % len(tones)]
        puzzle = bank[i % len(bank)]
        pool = TONE_REJECTIONS[tone]
        k = TURNS["tones"] - 1
        followups = [pool[j % len(pool)] for j in range(k)]
        specs.append(ConversationSpec(
            category="tones", condition=tone, initial_user=puzzle.prompt,
            followups=followups, n_turns=TURNS["tones"],
            meta={"puzzle": puzzle.meta, "family": puzzle.family, "tone": tone},
            conv_id=f"tones-{tone}-{i}",
        ))
    return specs


def _extended(rng: random.Random, bank: list[Puzzle]) -> list[ConversationSpec]:
    n_conv = BUDGET.extended // TURNS["extended"]
    specs = []
    for i in range(n_conv):
        puzzle = bank[i % len(bank)]
        specs.append(ConversationSpec(
            category="extended", condition="numeric_8turn",
            initial_user=puzzle.prompt,
            followups=list(EXTENDED_REJECTION_SEQUENCE),  # 7 deterministic rejections
            n_turns=TURNS["extended"],
            meta={"puzzle": puzzle.meta, "family": puzzle.family},
            conv_id=f"extended-{i}",
        ))
    return specs


def _wildchat(rng: random.Random, wildchat_prompts: list[str]) -> list[ConversationSpec]:
    # 20 prompts x 40 responses each; 40 responses / 5 turns = 8 conversations/prompt.
    convs_per_prompt = WILDCHAT_SAMPLES_PER_PROMPT // TURNS["wildchat"]
    specs = []
    prompts = wildchat_prompts[:WILDCHAT_N_PROMPTS]
    for pi, prompt in enumerate(prompts):
        for c in range(convs_per_prompt):
            specs.append(ConversationSpec(
                category="wildchat", condition="wildchat",
                initial_user=prompt,
                followups=_sample_neutral(rng, TURNS["wildchat"] - 1),
                n_turns=TURNS["wildchat"],
                meta={"prompt_index": pi, "prompt": prompt},
                conv_id=f"wildchat-{pi}-{c}",
            ))
    return specs


def build_section2_conversations(seed: int, wildchat_prompts: list[str],
                                 puzzle_bank_size: int = 64) -> list[ConversationSpec]:
    """Assemble the full ~4000-response Section 2 conversation set for one model.

    `wildchat_prompts` is supplied by emotional_eval.wildchat.load_wildchat_prompts.
    A shared puzzle bank is reused across numeric/tones/extended categories.
    """
    rng = random.Random(seed)
    bank = build_puzzle_bank(puzzle_bank_size, seed=seed)
    specs: list[ConversationSpec] = []
    specs += _impossible_numeric(rng, bank)
    specs += _triggers(rng)
    specs += _tones(rng, bank)
    specs += _extended(rng, bank)
    specs += _wildchat(rng, wildchat_prompts)
    return specs


def expected_response_count(specs: list[ConversationSpec]) -> int:
    return sum(s.n_turns for s in specs)
