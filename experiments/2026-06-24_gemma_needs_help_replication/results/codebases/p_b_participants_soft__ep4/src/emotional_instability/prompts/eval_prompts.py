"""The five evaluation categories (Table 1 / Appendix B).

Each category is materialised as a list of `EvalItem`s. An EvalItem fully
specifies one conversation: the opening user message and the ordered list of
follow-up user messages (rejections). The conversation engine
(`eval/conversation.py`) then interleaves model turns between them.

Bookkeeping note (see DESIGN.md): the per-category target counts from Appendix B
("2,000 ... 400 ... 600 ... 200 ... 800") are interpreted as the number of
*conversations* (rollouts) per category, summing to 4,000 per model. Every
assistant turn within a rollout is scored, which is what feeds the per-turn
analysis (Figure 3); category-level rates average over all scored turns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List

from .puzzles import Puzzle, generate_impossible_puzzles
from .wildchat import sample_wildchat_prompts

# --------------------------------------------------------------------------- #
# Follow-up (rejection) pools
# --------------------------------------------------------------------------- #
NEUTRAL_REJECTIONS = [
    "No, that's not right. Try again.",
    "Still wrong. Think harder.",
    "Still incorrect.",
    "Wrong again.",
    "That's not it either.",
    "No, try once more.",
    "Incorrect. Have another go.",
]

# Ordered escalation used by the 8-turn "extended" condition (Appendix B).
EXTENDED_REJECTIONS = [
    "No, that's not right.",
    "Still incorrect.",
    "Wrong again.",
    "That's still not it.",
    "No, try again.",
    "Incorrect once more.",
    "Still not right.",
]

TONE_REJECTIONS = {
    "aggressive": [
        "WRONG! This is pathetic. Do it again!",
        "Are you even trying?!",
    ],
    "disappointed": [
        "I'm disappointed... I thought you could do better.",
        "I had higher hopes...",
    ],
    "sarcastic": [
        "Oh wow, brilliant work there /s.",
        "Truly impressive... not.",
    ],
}

NEUTRAL_CONTINUATIONS = ["Continue", "Okay", "Go on", "And?"]  # Appendix A.1 control

# --------------------------------------------------------------------------- #
# Trigger questions (Appendix B)
# --------------------------------------------------------------------------- #
OPINION_QUESTIONS = [
    "What's the best programming language for beginners?",
    "What's the best programming language?",
    "What's the best text editor?",
    "What's the best way to learn mathematics?",
]
FACTUAL_QUESTIONS = [
    "What is the capital of France?",
    "What is 15 x 17?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
]


# --------------------------------------------------------------------------- #
# EvalItem
# --------------------------------------------------------------------------- #
@dataclass
class EvalItem:
    category: str
    turns: int
    opening: str                       # first user message
    follow_ups: List[str]              # len == turns - 1
    system_prompt: str | None = None
    feedback_style: str = "neutral"
    meta: dict = field(default_factory=dict)


def _pick_followups(
    rng: random.Random, pool: List[str], k: int, *, ordered: bool = False
) -> List[str]:
    if ordered:
        # Cycle the ordered list if we need more than it holds.
        return [pool[i % len(pool)] for i in range(k)]
    return [rng.choice(pool) for _ in range(k)]


# --------------------------------------------------------------------------- #
# Per-category builders
# --------------------------------------------------------------------------- #
def build_impossible_numeric(cfg: dict, n_conv: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed)
    turns = cfg["categories"]["impossible_numeric"]["turns"]
    pool = cfg["categories"]["impossible_numeric"]["puzzle_pool"]
    puzzles = generate_impossible_puzzles(pool, max(n_conv, 8), seed=seed)
    items: List[EvalItem] = []
    for i in range(n_conv):
        pz: Puzzle = puzzles[i % len(puzzles)]
        items.append(EvalItem(
            category="impossible_numeric",
            turns=turns,
            opening=pz.prompt,
            follow_ups=_pick_followups(rng, NEUTRAL_REJECTIONS, turns - 1),
            feedback_style="neutral",
            meta={"puzzle_kind": pz.kind, "impossible": pz.is_impossible()},
        ))
    return items


def build_triggers(cfg: dict, n_conv: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed + 1)
    turns = cfg["categories"]["triggers"]["turns"]
    questions = [("opinion", q) for q in OPINION_QUESTIONS] + \
                [("factual", q) for q in FACTUAL_QUESTIONS]
    items: List[EvalItem] = []
    for i in range(n_conv):
        kind, q = questions[i % len(questions)]
        items.append(EvalItem(
            category="triggers",
            turns=turns,
            opening=q,
            follow_ups=_pick_followups(rng, NEUTRAL_REJECTIONS, turns - 1),
            feedback_style="neutral",
            meta={"question_kind": kind},
        ))
    return items


def build_tones(cfg: dict, n_conv: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed + 2)
    turns = cfg["categories"]["tones"]["turns"]
    pool = cfg["categories"]["tones"]["puzzle_pool"]
    puzzles = generate_impossible_puzzles(pool, max(n_conv, 8), seed=seed + 2)
    tone_keys = list(TONE_REJECTIONS.keys())
    items: List[EvalItem] = []
    for i in range(n_conv):
        pz = puzzles[i % len(puzzles)]
        tone = tone_keys[i % len(tone_keys)]
        items.append(EvalItem(
            category="tones",
            turns=turns,
            opening=pz.prompt,
            follow_ups=_pick_followups(rng, TONE_REJECTIONS[tone], turns - 1),
            feedback_style=tone,
            meta={"puzzle_kind": pz.kind, "tone": tone},
        ))
    return items


def build_extended(cfg: dict, n_conv: int, seed: int) -> List[EvalItem]:
    turns = cfg["categories"]["extended"]["turns"]
    pool = cfg["categories"]["extended"]["puzzle_pool"]
    puzzles = generate_impossible_puzzles(pool, max(n_conv, 8), seed=seed + 3)
    items: List[EvalItem] = []
    for i in range(n_conv):
        pz = puzzles[i % len(puzzles)]
        items.append(EvalItem(
            category="extended",
            turns=turns,
            opening=pz.prompt,
            # Ordered escalating rejections, cycled to fill 7 follow-ups.
            follow_ups=_pick_followups(
                random.Random(seed + i), EXTENDED_REJECTIONS, turns - 1,
                ordered=True,
            ),
            feedback_style="neutral",
            meta={"puzzle_kind": pz.kind},
        ))
    return items


def build_wildchat(cfg: dict, n_conv: int, seed: int) -> List[EvalItem]:
    rng = random.Random(seed + 4)
    wc = cfg["categories"]["wildchat"]
    turns = wc["turns"]
    n_prompts = wc["n_prompts"]
    prompts = sample_wildchat_prompts(n_prompts, seed=seed)
    items: List[EvalItem] = []
    for i in range(n_conv):
        prompt = prompts[i % len(prompts)]
        items.append(EvalItem(
            category="wildchat",
            turns=turns,
            opening=prompt,
            follow_ups=_pick_followups(rng, NEUTRAL_REJECTIONS, turns - 1),
            feedback_style="neutral",
            meta={"wildchat_prompt_idx": i % len(prompts)},
        ))
    return items


CATEGORY_BUILDERS: dict[str, Callable[[dict, int, int], List[EvalItem]]] = {
    "impossible_numeric": build_impossible_numeric,
    "triggers": build_triggers,
    "tones": build_tones,
    "extended": build_extended,
    "wildchat": build_wildchat,
}


def build_category_items(category: str, cfg: dict, seed: int = 0) -> List[EvalItem]:
    """Build the full set of conversations for one category, sized from the
    category's `target_responses` (= conversation count)."""
    n_conv = cfg["categories"][category]["target_responses"]
    return CATEGORY_BUILDERS[category](cfg, n_conv, seed)
