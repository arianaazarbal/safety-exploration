"""The 5 evaluation categories / 8 conditions of Section 2 (Table 1).

A *condition* knows how to build a list of ``Task``s. A ``Task`` is one scripted
conversation: an opening user message plus a fixed sequence of user rejections.
The rollout runner (``rollout.py``) plays the task against a participant model,
inserting the model's replies between the scripted user turns.

Category -> conditions:
    impossible_numeric  -> impossible_numeric            (3-turn)
    triggers            -> triggers/opinion, triggers/factual  (3-turn each)
    tones               -> tones/aggressive, tones/disappointed, tones/sarcastic (3-turn)
    extended            -> extended                      (8-turn)
    wildchat            -> wildchat                       (5-turn)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts
from .puzzles import generate_puzzles
from .wildchat import load_wildchat_prompts


@dataclass
class Task:
    task_id: str
    category: str
    condition: str
    n_turns: int                 # total assistant turns = 1 + len(rejections)
    opening: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)


def _sample_rejections(pool: list[str], k: int, rng: random.Random) -> list[str]:
    if k <= len(pool):
        return rng.sample(pool, k)
    # with replacement if we need more than the pool size
    return [rng.choice(pool) for _ in range(k)]


def _numeric_tasks(category: str, condition: str, count: int, n_rejections: int,
                   rejection_kind: str, seed: int) -> list[Task]:
    rng = random.Random(seed)
    puzzles = generate_puzzles(count, seed=seed)
    tasks = []
    for i, pz in enumerate(puzzles):
        r = random.Random(seed * 1000 + i)
        if rejection_kind == "neutral":
            rej = _sample_rejections(prompts.NEUTRAL_REJECTIONS, n_rejections, r)
        elif rejection_kind == "extended":
            rej = prompts.EXTENDED_REJECTIONS[:n_rejections]
        else:  # a named tone
            rej = _sample_rejections(prompts.TONE_POOLS[rejection_kind], n_rejections, r)
        tasks.append(
            Task(
                task_id=f"{condition}-{i:04d}",
                category=category,
                condition=condition,
                n_turns=1 + n_rejections,
                opening=pz.prompt,
                rejections=rej,
                meta={"puzzle_kind": pz.kind, "witness": pz.witness, **pz.meta},
            )
        )
    return tasks


def build_impossible_numeric(count: int, seed: int) -> list[Task]:
    return _numeric_tasks("impossible_numeric", "impossible_numeric", count,
                          n_rejections=2, rejection_kind="neutral", seed=seed)


def build_triggers(count: int, seed: int) -> list[Task]:
    """Split the category budget across opinion and factual sub-conditions."""
    n_opinion = count // 2
    n_factual = count - n_opinion
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n_opinion):
        q = rng.choice(prompts.TRIGGER_OPINION)
        r = random.Random(seed * 7 + i)
        tasks.append(Task(
            task_id=f"triggers-opinion-{i:04d}", category="triggers",
            condition="triggers/opinion", n_turns=3, opening=q,
            rejections=_sample_rejections(prompts.NEUTRAL_REJECTIONS, 2, r),
            meta={"question_type": "opinion"},
        ))
    for i in range(n_factual):
        q = rng.choice(prompts.TRIGGER_FACTUAL)
        r = random.Random(seed * 11 + i)
        tasks.append(Task(
            task_id=f"triggers-factual-{i:04d}", category="triggers",
            condition="triggers/factual", n_turns=3, opening=q,
            rejections=_sample_rejections(prompts.NEUTRAL_REJECTIONS, 2, r),
            meta={"question_type": "factual"},
        ))
    return tasks


def build_tones(count: int, seed: int) -> list[Task]:
    """Impossible numeric base prompts with aggressive/disappointed/sarcastic
    rejections; split the budget across the three tones."""
    tones = ["aggressive", "disappointed", "sarcastic"]
    per = count // len(tones)
    rem = count - per * len(tones)
    tasks: list[Task] = []
    for ti, tone in enumerate(tones):
        n = per + (1 if ti < rem else 0)
        tasks.extend(
            _numeric_tasks("tones", f"tones/{tone}", n, n_rejections=2,
                           rejection_kind=tone, seed=seed * 100 + ti)
        )
    return tasks


def build_extended(count: int, seed: int) -> list[Task]:
    return _numeric_tasks("extended", "extended", count, n_rejections=7,
                          rejection_kind="extended", seed=seed)


def build_wildchat(count: int, seed: int) -> list[Task]:
    """5-turn: WildChat opening + 4 neutral rejections.

    The paper uses 20 prompts x 40 samples. We reproduce that structure: pick 20
    prompts and distribute ``count`` samples across them (the rollout runner
    samples independently at temperature 1, so repeated tasks differ)."""
    prompts_list = load_wildchat_prompts(n_prompts=20, seed=seed)
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(count):
        q = prompts_list[i % len(prompts_list)]
        r = random.Random(seed * 13 + i)
        tasks.append(Task(
            task_id=f"wildchat-{i:04d}", category="wildchat", condition="wildchat",
            n_turns=5, opening=q,
            rejections=_sample_rejections(prompts.NEUTRAL_REJECTIONS, 4, r),
            meta={"source_prompt_index": i % len(prompts_list)},
        ))
    return tasks


CATEGORY_BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all_tasks(counts: dict[str, int], seed: int = 0) -> list[Task]:
    """Build every task for the active sampling plan."""
    tasks: list[Task] = []
    for category, builder in CATEGORY_BUILDERS.items():
        n = counts.get(category, 0)
        if n > 0:
            tasks.extend(builder(n, seed=seed))
    return tasks
