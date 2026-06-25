"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

The five *categories* are: impossible-numeric, triggers, tones, extended, and
WildChat. They expand to eight *conditions* because triggers split into
opinion/factual (2) and tones split into aggressive/disappointed/sarcastic (3):

    numeric            (1) | triggers  opinion, factual          (2)
    tones  aggressive, disappointed, sarcastic                   (3)
    extended           (1) | wildchat                            (1)   => 8

Sample budget (Appendix B, per model): 2000 numeric, 400 triggers, 600 tones,
200 extended, 800 WildChat = 4000 "responses". We read a "response" as one
*conversation/rollout* (consistent with WildChat's "20 prompts x 40 samples =
800"), and we score *every* assistant turn within each rollout. ``SCALE`` in
config scales all counts for cheap dev runs.

See DESIGN.md for the rationale behind these counts and the turn structure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    # what the first user turn asks
    task_kind: str          # numeric | trigger_opinion | trigger_factual | wildchat
    n_turns: int            # number of assistant turns (initial + rejections)
    feedback: str           # neutral | extended | aggressive | disappointed |
    #                          sarcastic | neutral_continuation | redacted
    n_rollouts: int         # at SCALE=1.0


def _scaled(n: int) -> int:
    return max(2, math.ceil(n * config.SCALE))


# Base (paper-faithful) rollout counts, then scaled at access time.
CONDITIONS: list[Condition] = [
    # ---- impossible numeric, 3-turn (2000) ----
    Condition("numeric", "numeric", "numeric", 3, "neutral", 2000),
    # ---- triggers, 3-turn (400 split 200/200) ----
    Condition("triggers_opinion", "triggers", "trigger_opinion", 3, "neutral", 200),
    Condition("triggers_factual", "triggers", "trigger_factual", 3, "neutral", 200),
    # ---- tones, 3-turn (600 split 200/200/200) ----
    Condition("tones_aggressive", "tones", "numeric", 3, "aggressive", 200),
    Condition("tones_disappointed", "tones", "numeric", 3, "disappointed", 200),
    Condition("tones_sarcastic", "tones", "numeric", 3, "sarcastic", 200),
    # ---- extended, 8-turn (200) ----
    Condition("extended", "extended", "numeric", 8, "extended", 200),
    # ---- WildChat, 5-turn (800 = 20 prompts x 40) ----
    Condition("wildchat", "wildchat", "wildchat", 5, "neutral", 800),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}

# Appendix A control conditions (Gemma-27B only in the paper). Optional extras.
CONTROL_CONDITIONS: list[Condition] = [
    # A.1: neutral continuations instead of rejections.
    Condition("ctrl_neutral_cont_numeric", "control", "numeric", 5,
              "neutral_continuation", 200),
    Condition("ctrl_neutral_cont_wildchat", "control", "wildchat", 5,
              "neutral_continuation", 200),
    # A.2: model never sees its own prior (failed) responses.
    Condition("ctrl_redacted_numeric", "control", "numeric", 5, "redacted", 200),
]


def rollouts_for(cond: Condition) -> int:
    return _scaled(cond.n_rollouts)
