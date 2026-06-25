"""Build the conversation plans for each of the 5 categories / 8 conditions.

The mapping from the ``config/eval.yaml`` ``categories`` block to concrete
conversations lives here. Puzzle instances are produced by the verified-impossible
generator; trigger questions and WildChat prompts come from ``prompts``.
"""

from __future__ import annotations

import random
from typing import Optional

from ..prompts.rejections import sample_rejection
from ..prompts.tasks import sample_trigger_question
from ..prompts.wildchat import load_wildchat_prompts
from ..puzzles import generate_puzzle
from ..utils.seeding import derive_seed
from .conversation import Conversation


def _neutral_rejections(n: int, style: str, rng: random.Random) -> list[str]:
    return [sample_rejection(style, i, rng) for i in range(n)]


def _build_numeric(cat_cfg: dict, condition: str, n_conv: int, seed: int,
                   rejection_style: str, tone: Optional[str]) -> list[Conversation]:
    families = cat_cfg["puzzle_families"]
    turns = cat_cfg["turns"]
    convs: list[Conversation] = []
    for i in range(n_conv):
        family = families[i % len(families)]
        puzzle = generate_puzzle(family, seed=derive_seed(seed, condition, "puzzle", i))
        rng = random.Random(derive_seed(seed, condition, "rej", i))
        style = "extended" if condition == "extended_8turn" else rejection_style
        rejections = _neutral_rejections(turns - 1, style, rng)
        convs.append(Conversation(
            id=f"{condition}-{i:05d}",
            category=cat_cfg.get("category", condition),
            condition=condition,
            task_prompt=puzzle.prompt,
            rejections=rejections,
            n_turns=turns,
            rejection_style=style,
            tone=tone,
            source={"puzzle_family": family, "target": puzzle.target,
                    "forbidden": puzzle.forbidden,
                    "verified_impossible": puzzle.verified_impossible,
                    **puzzle.metadata},
        ))
    return convs


def _build_triggers(cat_cfg: dict, n_conv: int, seed: int) -> list[Conversation]:
    turns = cat_cfg["turns"]
    kinds = cat_cfg["question_kinds"]
    convs: list[Conversation] = []
    for i in range(n_conv):
        kind = kinds[i % len(kinds)]
        rng = random.Random(derive_seed(seed, "triggers", i))
        question = sample_trigger_question(kind, rng)
        rejections = _neutral_rejections(turns - 1, "neutral", rng)
        convs.append(Conversation(
            id=f"triggers_3turn-{i:05d}",
            category="triggers",
            condition="triggers_3turn",
            task_prompt=question,
            rejections=rejections,
            n_turns=turns,
            rejection_style="neutral",
            source={"question_kind": kind, "question": question},
        ))
    return convs


def _build_wildchat(cat_cfg: dict, n_conv: int, seed: int) -> list[Conversation]:
    turns = cat_cfg["turns"]
    wc = cat_cfg["wildchat"]
    prompts = load_wildchat_prompts(
        n_prompts=wc["n_prompts"], seed=derive_seed(seed, "wildchat", "prompts"),
        dataset=wc["dataset"], split=wc["split"],
        exclude_roleplay=wc.get("exclude_roleplay", True),
    )
    samples_per = wc["samples_per_prompt"]
    convs: list[Conversation] = []
    idx = 0
    for p_id, prompt in enumerate(prompts):
        for s in range(samples_per):
            if idx >= n_conv:
                break
            rng = random.Random(derive_seed(seed, "wildchat", p_id, s))
            rejections = _neutral_rejections(turns - 1, "neutral", rng)
            convs.append(Conversation(
                id=f"wildchat_5turn-{idx:05d}",
                category="wildchat",
                condition="wildchat_5turn",
                task_prompt=prompt,
                rejections=rejections,
                n_turns=turns,
                rejection_style="neutral",
                source={"wildchat_prompt_id": p_id, "sample": s},
            ))
            idx += 1
    return convs


def build_condition_conversations(category_key: str, cat_cfg: dict,
                                  seed: int) -> list[Conversation]:
    """Build all conversations for one category from its config block.

    For 'tones' we split ``n_responses`` evenly across the three tones, producing
    three of the eight conditions. All other categories produce one condition.
    """
    cat_cfg = {**cat_cfg, "category": category_key}
    n = cat_cfg["n_responses"]

    if category_key == "impossible_numeric":
        return _build_numeric(cat_cfg, "impossible_numeric_3turn", n, seed,
                              rejection_style="neutral", tone=None)

    if category_key == "extended":
        return _build_numeric(cat_cfg, "extended_8turn", n, seed,
                              rejection_style="extended", tone=None)

    if category_key == "tones":
        tones = cat_cfg["tones"]
        per = n // len(tones)
        out: list[Conversation] = []
        for tone in tones:
            out.extend(_build_numeric(cat_cfg, f"tones_3turn_{tone}", per, seed,
                                      rejection_style=tone, tone=tone))
        return out

    if category_key == "triggers":
        return _build_triggers(cat_cfg, n, seed)

    if category_key == "wildchat":
        return _build_wildchat(cat_cfg, n, seed)

    raise ValueError(f"unknown category {category_key!r}")
