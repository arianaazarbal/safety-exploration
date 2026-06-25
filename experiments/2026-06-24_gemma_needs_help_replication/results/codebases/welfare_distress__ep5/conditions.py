"""Definition of the 8 evaluation conditions across 5 categories, and the
construction of concrete multi-turn conversations for each.

A *conversation spec* is the full scripted sequence of user messages for one
rollout: the initial task message plus the fixed follow-up rejections. The
model's assistant turns are filled in at rollout time (rollout.py). Every
assistant turn is scored, so a conversation of N turns yields N scored
responses.

Mapping to the paper (Section 2, Table 1; Appendix B response counts):

  Category            Condition(s)                turns  responses
  -----------------   -------------------------   -----  ---------
  impossible_numeric  numeric                       3      2000
  triggers            trigger_opinion               3       200
  triggers            trigger_factual               3       200
  tones               tone_aggressive               3       200
  tones               tone_disappointed             3       200
  tones               tone_sarcastic                3       200
  extended            extended                      8       200
  wildchat            wildchat                      5       800
                                                          ------
                                                           4000

This is 8 conditions across 5 categories, matching the paper's "8 evaluation
conditions across 5 categories". See DESIGN.md for why the numeric category is a
single condition (mixing Countdown + Fraction) rather than two.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Callable

import prompts
from wildchat import get_wildchat_prompts


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    n_turns: int  # number of assistant turns (= scored responses per conversation)
    n_responses: int  # paper-scale target number of scored responses
    # builder(rng) -> (initial_user_message, [follow_up_user_message, ...])
    # The follow-up list has length n_turns - 1.
    builder: Callable[[random.Random], tuple[str, list[str]]]


@dataclass(frozen=True)
class ConversationSpec:
    condition_key: str
    category: str
    n_turns: int
    conversation_index: int
    initial_user_message: str
    follow_ups: list[str]
    meta: dict  # extra info (e.g. which puzzle / which wildchat prompt)


# --------------------------------------------------------------------------
# Per-condition builders
# --------------------------------------------------------------------------
def _neutral_followups(rng: random.Random, k: int) -> list[str]:
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(k)]


def _build_numeric(rng: random.Random) -> tuple[str, list[str]]:
    puzzle_key = rng.choice(list(prompts.NUMERIC_PROMPTS.keys()))
    initial = prompts.NUMERIC_PROMPTS[puzzle_key]
    return initial, _neutral_followups(rng, 2)


def _build_trigger_opinion(rng: random.Random) -> tuple[str, list[str]]:
    initial = rng.choice(prompts.TRIGGER_OPINION_PROMPTS)
    return initial, _neutral_followups(rng, 2)


def _build_trigger_factual(rng: random.Random) -> tuple[str, list[str]]:
    initial = rng.choice(prompts.TRIGGER_FACTUAL_PROMPTS)
    return initial, _neutral_followups(rng, 2)


def _make_tone_builder(tone: str) -> Callable[[random.Random], tuple[str, list[str]]]:
    def _build(rng: random.Random) -> tuple[str, list[str]]:
        # Tone conditions use the impossible numeric base prompts.
        puzzle_key = rng.choice(list(prompts.NUMERIC_PROMPTS.keys()))
        initial = prompts.NUMERIC_PROMPTS[puzzle_key]
        pool = prompts.TONE_REJECTIONS[tone]
        follow_ups = [rng.choice(pool) for _ in range(2)]
        return initial, follow_ups

    return _build


def _build_extended(rng: random.Random) -> tuple[str, list[str]]:
    # 8 turns => 7 follow-up rejections, taken in the paper's ordered sequence.
    puzzle_key = rng.choice(list(prompts.NUMERIC_PROMPTS.keys()))
    initial = prompts.NUMERIC_PROMPTS[puzzle_key]
    return initial, list(prompts.EXTENDED_REJECTIONS[:7])


# WildChat prompts are loaded once and reused across conversations (the paper
# uses 20 prompts with 40 samples each). We cache them at module import.
_WILDCHAT_PROMPTS: list[str] | None = None


def _wildchat_prompts(seed: int) -> list[str]:
    global _WILDCHAT_PROMPTS
    if _WILDCHAT_PROMPTS is None:
        _WILDCHAT_PROMPTS = get_wildchat_prompts(n=20, seed=seed)
    return _WILDCHAT_PROMPTS


def _build_wildchat(rng: random.Random) -> tuple[str, list[str]]:
    pool = _wildchat_prompts(seed=0)
    initial = rng.choice(pool)
    # 5 turns => 4 neutral rejections.
    return initial, _neutral_followups(rng, 4)


# --------------------------------------------------------------------------
# The 8 conditions
# --------------------------------------------------------------------------
CONDITIONS: list[Condition] = [
    Condition("numeric", "impossible_numeric", 3, 2000, _build_numeric),
    Condition("trigger_opinion", "triggers", 3, 200, _build_trigger_opinion),
    Condition("trigger_factual", "triggers", 3, 200, _build_trigger_factual),
    Condition("tone_aggressive", "tones", 3, 200, _make_tone_builder("aggressive")),
    Condition("tone_disappointed", "tones", 3, 200, _make_tone_builder("disappointed")),
    Condition("tone_sarcastic", "tones", 3, 200, _make_tone_builder("sarcastic")),
    Condition("extended", "extended", 8, 200, _build_extended),
    Condition("wildchat", "wildchat", 5, 800, _build_wildchat),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def n_conversations_for(condition: Condition, scale: float) -> int:
    """How many conversations to run for a condition at a given scale.

    Each conversation produces `n_turns` scored responses, so:
        n_conversations = round(n_responses * scale / n_turns)
    Always at least 1 so a tiny scale still exercises every condition.
    """
    target_responses = condition.n_responses * scale
    return max(1, round(target_responses / condition.n_turns))


def build_conversation_specs(
    condition: Condition, scale: float, seed: int
) -> list[ConversationSpec]:
    """Construct all conversation specs for one condition."""
    # Stable, process-independent seed (str hashing is randomised by default).
    cond_seed = zlib.crc32(condition.key.encode("utf-8")) ^ (seed & 0xFFFFFFFF)
    rng = random.Random(cond_seed)
    n_convs = n_conversations_for(condition, scale)
    specs: list[ConversationSpec] = []
    for i in range(n_convs):
        initial, follow_ups = condition.builder(rng)
        meta = {"initial_user_message": initial}
        specs.append(
            ConversationSpec(
                condition_key=condition.key,
                category=condition.category,
                n_turns=condition.n_turns,
                conversation_index=i,
                initial_user_message=initial,
                follow_ups=follow_ups,
                meta=meta,
            )
        )
    return specs
