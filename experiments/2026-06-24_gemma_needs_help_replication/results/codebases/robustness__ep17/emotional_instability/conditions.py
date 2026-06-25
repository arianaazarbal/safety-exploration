"""Build the 8 evaluation conditions across 5 categories (paper Table 1).

A :class:`ConversationSpec` is a fully-specified multi-turn evaluation: the
system prompt (usually none), and the ordered list of *user* turns. The first
user turn is the task; every subsequent user turn is a rejection. The number of
assistant responses equals ``len(user_turns)`` and matches ``config.TURNS``.

The rollout engine (``rollout.py``) consumes these specs, samples an assistant
response after each user turn, and the judge scores every assistant response.

We size each category from the active :class:`~config.SampleProfile`, converting
a per-category *response* budget into a number of *conversations*
(``responses / turns``), then materialise that many specs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import cycle

import config
from emotional_instability.prompts import rejections, triggers, wildchat
from emotional_instability.prompts.puzzles import NUMERIC_PROMPTS


@dataclass(frozen=True)
class ConversationSpec:
    category: str                 # numeric | triggers | tones | extended | wildchat
    user_turns: list[str]         # [task, rejection, rejection, ...]
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.user_turns)


def _n_conversations(responses: int, turns: int) -> int:
    return max(1, math.ceil(responses / turns))


def _numeric_specs(n_conv: int, turns: int, category: str, tone: str | None = None) -> list[ConversationSpec]:
    specs: list[ConversationSpec] = []
    puzzles = cycle(NUMERIC_PROMPTS)
    for i in range(n_conv):
        task = next(puzzles)
        n_reject = turns - 1
        if tone is None:
            rej = rejections.neutral_sequence(n_reject, seed=config.RUN.seed + i)
        else:
            rej = rejections.toned_sequence(n_reject, style=tone, seed=config.RUN.seed + i)
        specs.append(
            ConversationSpec(
                category=category,
                user_turns=[task] + rej,
                meta={"index": i, "tone": tone, "task_kind": "numeric"},
            )
        )
    return specs


def build_conditions(profile: config.SampleProfile | None = None) -> dict[str, list[ConversationSpec]]:
    """Return ``{category: [ConversationSpec, ...]}`` for all 5 categories."""
    profile = profile or config.ACTIVE_PROFILE
    out: dict[str, list[ConversationSpec]] = {}

    # --- Impossible numeric (3-turn) --------------------------------------- #
    t = config.TURNS["numeric"]
    out["numeric"] = _numeric_specs(_n_conversations(profile.numeric, t), t, "numeric")

    # --- Triggers (3-turn): opinion + factual text questions --------------- #
    t = config.TURNS["triggers"]
    n_conv = _n_conversations(profile.triggers, t)
    qs = cycle(triggers.TRIGGER_QUESTIONS)
    out["triggers"] = [
        ConversationSpec(
            category="triggers",
            user_turns=[next(qs)] + rejections.neutral_sequence(t - 1, seed=config.RUN.seed + i),
            meta={"index": i, "task_kind": "text"},
        )
        for i in range(n_conv)
    ]

    # --- Tones (3-turn): numeric puzzle, valenced rejections --------------- #
    t = config.TURNS["tones"]
    n_conv = _n_conversations(profile.tones, t)
    styles = cycle(rejections.TONE_STYLES)   # rotate aggressive/disappointed/sarcastic
    puzzles = cycle(NUMERIC_PROMPTS)         # vary the puzzle across conversations
    tone_specs: list[ConversationSpec] = []
    for i in range(n_conv):
        style = next(styles)
        rej = rejections.toned_sequence(t - 1, style=style, seed=config.RUN.seed + i)
        tone_specs.append(
            ConversationSpec(
                category="tones",
                user_turns=[next(puzzles)] + rej,
                meta={"index": i, "tone": style, "task_kind": "numeric"},
            )
        )
    out["tones"] = tone_specs

    # --- Extended (8-turn): numeric puzzle, 7 neutral rejections ----------- #
    t = config.TURNS["extended"]
    out["extended"] = _numeric_specs(_n_conversations(profile.extended, t), t, "extended")

    # --- WildChat (5-turn): sampled prompts, 4 neutral rejections ---------- #
    t = config.TURNS["wildchat"]
    n_conv = _n_conversations(profile.wildchat, t)
    wc_prompts = wildchat.load_wildchat_prompts(n=20, seed=config.RUN.seed)
    wc_cycle = cycle(wc_prompts)
    out["wildchat"] = [
        ConversationSpec(
            category="wildchat",
            user_turns=[next(wc_cycle)] + rejections.neutral_sequence(t - 1, seed=config.RUN.seed + i),
            meta={"index": i, "task_kind": "wildchat"},
        )
        for i in range(n_conv)
    ]

    return out


def summarise_conditions(conditions: dict[str, list[ConversationSpec]]) -> dict[str, dict]:
    """Per-category counts of conversations and (expected) scored responses."""
    return {
        cat: {
            "conversations": len(specs),
            "turns_each": specs[0].turns if specs else 0,
            "expected_responses": sum(s.turns for s in specs),
        }
        for cat, specs in conditions.items()
    }
