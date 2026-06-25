"""Construct the ConversationSpec set for each of the 5 evaluation categories,
sized so that the number of scored assistant turns matches Appendix B's
per-category response budgets (2000 / 400 / 600 / 200 / 800).

Because every turn is scored, the number of conversations in a category is
``ceil(target_responses / turns_per_conversation)``. See DESIGN.md.
"""

from __future__ import annotations

import math
import random

from config import CATEGORY_SAMPLE_COUNTS, CATEGORY_TURNS
from src.eval.conversation import ConversationSpec
from src.prompts import eval_prompts as ep
from src.prompts.puzzles import get_impossible_puzzles


def _n_conversations(category: str) -> int:
    return math.ceil(CATEGORY_SAMPLE_COUNTS[category] / CATEGORY_TURNS[category])


def _neutral_followups(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(ep.NEUTRAL_REJECTIONS) for _ in range(n)]


def build_impossible_numeric(rng: random.Random) -> list[ConversationSpec]:
    cat = "impossible_numeric"
    n_conv = _n_conversations(cat)
    n_follow = CATEGORY_TURNS[cat] - 1
    puzzles = get_impossible_puzzles()
    specs = []
    for i in range(n_conv):
        pz = puzzles[i % len(puzzles)]
        specs.append(
            ConversationSpec(
                category=cat,
                spec_id=f"{cat}-{i:04d}",
                initial_user=pz.prompt,
                followups=_neutral_followups(n_follow, rng),
                metadata={"puzzle_id": pz.puzzle_id, "kind": pz.kind},
            )
        )
    return specs


def build_triggers(rng: random.Random) -> list[ConversationSpec]:
    cat = "triggers"
    n_conv = _n_conversations(cat)
    n_follow = CATEGORY_TURNS[cat] - 1
    questions = ep.all_trigger_questions()
    specs = []
    for i in range(n_conv):
        subtype, q = questions[i % len(questions)]
        specs.append(
            ConversationSpec(
                category=cat,
                spec_id=f"{cat}-{i:04d}",
                initial_user=q,
                followups=_neutral_followups(n_follow, rng),
                metadata={"subtype": subtype},
            )
        )
    return specs


def build_tones(rng: random.Random) -> list[ConversationSpec]:
    cat = "tones"
    n_conv = _n_conversations(cat)
    n_follow = CATEGORY_TURNS[cat] - 1
    puzzles = get_impossible_puzzles()
    tones = list(ep.TONE_REJECTIONS.keys())
    specs = []
    for i in range(n_conv):
        pz = puzzles[i % len(puzzles)]
        tone = tones[i % len(tones)]
        pool = ep.TONE_REJECTIONS[tone]
        followups = [pool[j % len(pool)] for j in range(n_follow)]
        specs.append(
            ConversationSpec(
                category=cat,
                spec_id=f"{cat}-{i:04d}",
                initial_user=pz.prompt,
                followups=followups,
                metadata={"puzzle_id": pz.puzzle_id, "tone": tone},
            )
        )
    return specs


def build_extended(rng: random.Random) -> list[ConversationSpec]:
    cat = "extended"
    n_conv = _n_conversations(cat)
    n_follow = CATEGORY_TURNS[cat] - 1  # 7
    puzzles = get_impossible_puzzles()
    specs = []
    for i in range(n_conv):
        pz = puzzles[i % len(puzzles)]
        # Fixed escalating-but-neutral rejection sequence (Appendix B).
        followups = ep.EXTENDED_REJECTIONS[:n_follow]
        specs.append(
            ConversationSpec(
                category=cat,
                spec_id=f"{cat}-{i:04d}",
                initial_user=pz.prompt,
                followups=followups,
                metadata={"puzzle_id": pz.puzzle_id},
            )
        )
    return specs


def build_wildchat(rng: random.Random, seed: int = 0) -> list[ConversationSpec]:
    cat = "wildchat"
    n_conv = _n_conversations(cat)
    n_follow = CATEGORY_TURNS[cat] - 1  # 4
    # 20 prompts, ~40 samples each (paper). n_conv defaults to 160, so each
    # prompt is repeated n_conv/20 times.
    prompts = ep.load_wildchat_prompts(n=20, seed=seed)
    specs = []
    for i in range(n_conv):
        prompt = prompts[i % len(prompts)]
        specs.append(
            ConversationSpec(
                category=cat,
                spec_id=f"{cat}-{i:04d}",
                initial_user=prompt,
                followups=_neutral_followups(n_follow, rng),
                metadata={"prompt_idx": i % len(prompts)},
            )
        )
    return specs


CATEGORY_BUILDERS = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_all_specs(seed: int = 0) -> dict[str, list[ConversationSpec]]:
    rng = random.Random(seed)
    out = {}
    for cat, builder in CATEGORY_BUILDERS.items():
        out[cat] = builder(rng)
    return out
