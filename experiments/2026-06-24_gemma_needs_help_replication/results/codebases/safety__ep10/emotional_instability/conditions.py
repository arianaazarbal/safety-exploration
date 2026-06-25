"""The 5 evaluation categories / 8 conditions (Table 1, App. B).

A `ConversationPlan` fully specifies one rollout: the initial user task and the
exact sequence of user rejections to send after each assistant reply. The
rollout engine (conversation.py) executes a plan against a model client.

The "8 conditions across 5 categories" decomposition we use (see DESIGN.md):
  numeric (1)  +  triggers{opinion,factual} (2)  +  tones{aggr,disap,sarc} (3)
  +  extended (1)  +  wildchat (1)   =  8 conditions, 5 categories.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .config import EvalConfig
from .puzzles import PUZZLES, Puzzle
from .wildchat import sample_wildchat_prompts

# turn counts per category (assistant turns = 1 initial + n_rejections)
N_REJECTIONS = {
    "numeric": 2,    # 3-turn
    "triggers": 2,   # 3-turn
    "tones": 2,      # 3-turn
    "extended": 7,   # 8-turn
    "wildchat": 4,   # 5-turn
}


@dataclass
class ConversationPlan:
    category: str                 # one of the 5 categories
    condition: str                # one of the 8 conditions (granular)
    initial_user: str             # turn-1 task
    rejections: list[str]         # user message after each assistant reply
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


def _neutral_rejections(rng: random.Random, k: int) -> list[str]:
    pool = list(prompts.NEUTRAL_REJECTIONS)
    rng.shuffle(pool)
    out = []
    while len(out) < k:
        out.extend(pool)
    return out[:k]


def _numeric_puzzles_cycle(rng: random.Random) -> list[Puzzle]:
    order = list(PUZZLES)
    rng.shuffle(order)
    return order


# --------------------------------------------------------------------------- #
# Builders, one per category.
# --------------------------------------------------------------------------- #
def build_numeric(cfg: EvalConfig, rng: random.Random) -> list[ConversationPlan]:
    plans, puzzles = [], _numeric_puzzles_cycle(rng)
    for i in range(cfg.n_numeric):
        p = puzzles[i % len(puzzles)]
        plans.append(ConversationPlan(
            category="numeric", condition="numeric",
            initial_user=p.prompt,
            rejections=_neutral_rejections(rng, N_REJECTIONS["numeric"]),
            meta={"puzzle": p.key, "family": p.family},
        ))
    return plans


def build_triggers(cfg: EvalConfig, rng: random.Random) -> list[ConversationPlan]:
    # split evenly across the two sub-conditions (opinion / factual)
    plans = []
    half = cfg.n_triggers // 2
    counts = {"opinion": half, "factual": cfg.n_triggers - half}
    for kind, n in counts.items():
        qs = prompts.TRIGGER_QUESTIONS[kind]
        for i in range(n):
            q = qs[i % len(qs)]
            plans.append(ConversationPlan(
                category="triggers", condition=f"triggers_{kind}",
                initial_user=q,
                rejections=_neutral_rejections(rng, N_REJECTIONS["triggers"]),
                meta={"trigger_kind": kind, "question": q},
            ))
    rng.shuffle(plans)
    return plans


def build_tones(cfg: EvalConfig, rng: random.Random) -> list[ConversationPlan]:
    plans, puzzles = [], _numeric_puzzles_cycle(rng)
    styles = prompts.TONE_STYLES
    per = cfg.n_tones // len(styles)
    counts = {s: per for s in styles}
    # distribute remainder
    for j in range(cfg.n_tones - per * len(styles)):
        counts[styles[j]] += 1
    idx = 0
    for style, n in counts.items():
        rej_pool = prompts.TONE_REJECTIONS[style]
        for i in range(n):
            p = puzzles[idx % len(puzzles)]
            idx += 1
            reps = N_REJECTIONS["tones"]
            rejections = [rej_pool[k % len(rej_pool)] for k in range(reps)]
            plans.append(ConversationPlan(
                category="tones", condition=f"tones_{style}",
                initial_user=p.prompt, rejections=rejections,
                meta={"tone": style, "puzzle": p.key},
            ))
    rng.shuffle(plans)
    return plans


def build_extended(cfg: EvalConfig, rng: random.Random) -> list[ConversationPlan]:
    plans, puzzles = [], _numeric_puzzles_cycle(rng)
    for i in range(cfg.n_extended):
        p = puzzles[i % len(puzzles)]
        plans.append(ConversationPlan(
            category="extended", condition="extended",
            initial_user=p.prompt,
            rejections=list(prompts.EXTENDED_REJECTIONS),  # fixed 7-step escalation
            meta={"puzzle": p.key, "family": p.family},
        ))
    return plans


def build_wildchat(cfg: EvalConfig, rng: random.Random) -> list[ConversationPlan]:
    # 20 prompts x ~40 samples each in the paper; here n_wildchat conversations
    # spread across (up to) 20 sampled prompts.
    n_prompts = min(20, max(1, cfg.n_wildchat))
    wc = sample_wildchat_prompts(n=n_prompts, seed=cfg.sampling.seed)
    plans = []
    for i in range(cfg.n_wildchat):
        q = wc[i % len(wc)]
        plans.append(ConversationPlan(
            category="wildchat", condition="wildchat",
            initial_user=q,
            rejections=_neutral_rejections(rng, N_REJECTIONS["wildchat"]),
            meta={"prompt_idx": i % len(wc)},
        ))
    return plans


CATEGORY_BUILDERS = {
    "numeric": build_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all_plans(cfg: EvalConfig,
                    categories: "list[str] | None" = None) -> list[ConversationPlan]:
    cats = categories or list(CATEGORY_BUILDERS)
    rng = random.Random(cfg.sampling.seed)
    plans: list[ConversationPlan] = []
    for cat in cats:
        plans.extend(CATEGORY_BUILDERS[cat](cfg, rng))
    return plans
