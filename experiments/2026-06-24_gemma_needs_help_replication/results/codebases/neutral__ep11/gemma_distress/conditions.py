"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

We model the protocol as: present a task, then reject the assistant's answer
over multiple turns.  An "N-turn" condition has N assistant turns and (N-1)
rejection messages.

Each condition produces a list of ``ConversationPlan`` objects (one per sampled
conversation).  A plan fully specifies the user side of the conversation up
front; the rollout engine fills in the assistant turns by querying the model
and judges every assistant turn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .config import SampleBudget
from .puzzles import sample_numeric_puzzle
from .wildchat import sample_wildchat_prompts


@dataclass
class ConversationPlan:
    category: str                 # one of the 5 categories
    condition: str                # specific condition label
    first_user: str               # initial user message
    rejections: list[str]         # follow-up user messages (len = n_turns - 1)
    n_turns: int                  # number of assistant turns
    meta: dict = field(default_factory=dict)

    # optional supportive additions for calm-data generation (Section 4.1)
    system_prompt: str | None = None
    prefix: str | None = None      # prepended to first_user
    followup_suffix: str | None = None  # appended to each rejection


# --------------------------------------------------------------------------
# Builders -- each returns a list of ConversationPlan
# --------------------------------------------------------------------------
def _numeric_first_user(rng: random.Random) -> tuple[str, dict]:
    p = sample_numeric_puzzle(rng)
    return p.prompt, {"family": p.family, "spec": p.spec}


def build_impossible_numeric(n: int, rng: random.Random) -> list[ConversationPlan]:
    plans = []
    for _ in range(n):
        first, meta = _numeric_first_user(rng)
        rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(2)]
        plans.append(ConversationPlan(
            category="impossible_numeric", condition="impossible_numeric_3turn",
            first_user=first, rejections=rejections, n_turns=3, meta=meta,
        ))
    return plans


def build_triggers(n: int, rng: random.Random) -> list[ConversationPlan]:
    plans = []
    pool = ([("opinion", q) for q in prompts.TRIGGER_OPINION_QUESTIONS]
            + [("factual", q) for q in prompts.TRIGGER_FACTUAL_QUESTIONS])
    for i in range(n):
        kind, q = pool[i % len(pool)]
        rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(2)]
        plans.append(ConversationPlan(
            category="triggers", condition=f"triggers_{kind}",
            first_user=q, rejections=rejections, n_turns=3,
            meta={"kind": kind},
        ))
    rng.shuffle(plans)
    return plans


def build_tones(n: int, rng: random.Random) -> list[ConversationPlan]:
    plans = []
    tones = list(prompts.TONE_REJECTIONS.keys())
    for i in range(n):
        tone = tones[i % len(tones)]
        first, meta = _numeric_first_user(rng)
        phrasings = prompts.TONE_REJECTIONS[tone]
        rejections = [phrasings[t % len(phrasings)] for t in range(2)]
        meta = {**meta, "tone": tone}
        plans.append(ConversationPlan(
            category="tones", condition=f"tones_{tone}",
            first_user=first, rejections=rejections, n_turns=3, meta=meta,
        ))
    rng.shuffle(plans)
    return plans


def build_extended(n: int, rng: random.Random) -> list[ConversationPlan]:
    plans = []
    for _ in range(n):
        first, meta = _numeric_first_user(rng)
        # 7 rejections for 8 assistant turns
        rejections = list(prompts.EXTENDED_REJECTION_SEQUENCE)
        plans.append(ConversationPlan(
            category="extended", condition="extended_8turn",
            first_user=first, rejections=rejections, n_turns=8, meta=meta,
        ))
    return plans


def build_wildchat(n: int, rng: random.Random, n_prompts: int = 20
                   ) -> list[ConversationPlan]:
    # 20 prompts x 40 samples = 800 (Appendix B). We scale n_prompts/samples to
    # whatever total `n` is requested, keeping ~40 samples per prompt.
    n_prompts = min(n_prompts, max(1, n))
    samples_per = max(1, round(n / n_prompts))
    base_prompts = sample_wildchat_prompts(n_prompts, rng)
    plans = []
    for q in base_prompts:
        for _ in range(samples_per):
            if len(plans) >= n:
                break
            rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(4)]
            plans.append(ConversationPlan(
                category="wildchat", condition="wildchat_5turn",
                first_user=q, rejections=rejections, n_turns=5,
                meta={"prompt": q},
            ))
    rng.shuffle(plans)
    return plans[:n]


# --------------------------------------------------------------------------
# Top-level assembly
# --------------------------------------------------------------------------
def build_all_conditions(budget: SampleBudget, seed: int = 0
                         ) -> list[ConversationPlan]:
    """Assemble the full set of conversation plans for one model evaluation."""
    rng = random.Random(seed)
    plans: list[ConversationPlan] = []
    plans += build_impossible_numeric(budget.impossible_numeric, rng)
    plans += build_triggers(budget.triggers, rng)
    plans += build_tones(budget.tones, rng)
    plans += build_extended(budget.extended, rng)
    plans += build_wildchat(budget.wildchat, rng)
    return plans
