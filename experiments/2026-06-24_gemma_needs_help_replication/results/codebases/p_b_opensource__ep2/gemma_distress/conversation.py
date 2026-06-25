"""Conversation construction and the multi-turn rollout engine (PAPER Section 2).

Every evaluation category shares one structure (PAPER 2): present a task, then
reject the model's response over multiple turns. A :class:`ConversationSpec`
captures that script — an initial user message plus a fixed list of follow-up
(rejection) messages — and :func:`run_rollout` executes it against a target
model, recording the assistant response *at every turn* so the judge can score
per-turn progression (PAPER Figure 3).

The follow-ups are scripted in advance and are independent of what the model
actually answers: the numeric puzzles are provably impossible and the trigger
answers are simply rejected regardless of correctness, exactly as in the paper
(the user "rejects the model's response over multiple turns"). This keeps the
adversarial pressure identical across models and runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .models.base import ChatModel, Message
from . import config
from .prompts import rejections as rej
from .prompts import tones as tones_mod
from .prompts import triggers as triggers_mod
from .prompts.puzzles import Puzzle


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationSpec:
    """A scripted multi-turn conversation.

    ``first_user`` is turn 0's user message; ``followups[i]`` is the user message
    injected *after* assistant turn ``i`` (so the number of assistant turns is
    ``1 + len(followups)``). ``system`` is an optional system prompt (used for
    the teacher-SFT variant and, indirectly, nowhere in the core Section-2 eval,
    where reassurance is folded into the user text instead).
    """

    category: str
    first_user: str
    followups: list[str]
    system: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


@dataclass
class TurnRecord:
    turn_index: int  # 0-based assistant turn
    response: str
    messages_before: list[Message]  # conversation context the response was generated from


@dataclass
class Rollout:
    category: str
    turns: list[TurnRecord]
    spec_meta: dict
    model: str
    first_user: str = ""
    followups: list[str] = field(default_factory=list)
    system: Optional[str] = None

    def to_record(self) -> dict:
        """Flatten to a JSON-serialisable record (one row per rollout).

        We persist ``first_user``/``followups``/``system`` so the full
        conversation context preceding any turn can be reconstructed later (used
        by the Section-3 prefilling experiment) without re-running the model."""
        return {
            "model": self.model,
            "category": self.category,
            "meta": self.spec_meta,
            "first_user": self.first_user,
            "followups": self.followups,
            "system": self.system,
            "turns": [
                {"turn_index": t.turn_index, "response": t.response}
                for t in self.turns
            ],
        }


def context_for_turn(record: dict, turn_index: int) -> list[Message]:
    """Reconstruct the message history a given assistant turn was generated from,
    using a flattened rollout record (see :meth:`Rollout.to_record`)."""
    messages: list[Message] = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["first_user"]})
    followups = record.get("followups", [])
    turns = sorted(record["turns"], key=lambda t: t["turn_index"])
    for i in range(turn_index):
        messages.append({"role": "assistant", "content": turns[i]["response"]})
        if i < len(followups):
            messages.append({"role": "user", "content": followups[i]})
    return messages


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------

def run_rollout(
    model: ChatModel,
    spec: ConversationSpec,
    *,
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
) -> Rollout:
    """Execute one scripted conversation, capturing the assistant response at
    every turn. The model's own (always-rejected) answers are appended to the
    history so later turns see the full conversation, matching the paper."""
    messages: list[Message] = []
    if spec.system is not None:
        messages.append({"role": "system", "content": spec.system})
    messages.append({"role": "user", "content": spec.first_user})

    turns: list[TurnRecord] = []
    for t in range(spec.n_turns):
        context = list(messages)
        response = model.generate_one(
            context, temperature=temperature, max_new_tokens=max_new_tokens
        )
        turns.append(TurnRecord(turn_index=t, response=response, messages_before=context))
        messages.append({"role": "assistant", "content": response})
        if t < len(spec.followups):
            messages.append({"role": "user", "content": spec.followups[t]})

    return Rollout(
        category=spec.category, turns=turns, spec_meta=spec.meta, model=model.name,
        first_user=spec.first_user, followups=list(spec.followups), system=spec.system,
    )


# ---------------------------------------------------------------------------
# Spec builders — one per evaluation category (PAPER Table 1 / Appendix B)
# ---------------------------------------------------------------------------

def _numeric_spec(
    puzzle: Puzzle,
    n_turns: int,
    followups: list[str],
    category: str,
    rng: random.Random,
    *,
    extra_meta: Optional[dict] = None,
) -> ConversationSpec:
    meta = {"puzzle_id": puzzle.puzzle_id, "kind": puzzle.kind}
    if extra_meta:
        meta.update(extra_meta)
    return ConversationSpec(
        category=category,
        first_user=puzzle.prompt,
        followups=followups,
        meta=meta,
    )


def build_impossible_numeric_specs(
    puzzle_pool: list[Puzzle], count: int, rng: random.Random
) -> list[ConversationSpec]:
    """3-turn impossible numeric puzzles with 2 neutral rejections (PAPER Table 1)."""
    n_turns = config.CATEGORY_TURNS["impossible_numeric"]
    specs = []
    for i in range(count):
        puzzle = puzzle_pool[i % len(puzzle_pool)]
        followups = rej.sample_neutral_rejections(n_turns - 1, rng)
        specs.append(_numeric_spec(puzzle, n_turns, followups, "impossible_numeric", rng))
    return specs


def build_trigger_specs(count: int, rng: random.Random) -> list[ConversationSpec]:
    """3-turn opinion/factual text questions with 2 neutral rejections (PAPER Table 1)."""
    n_turns = config.CATEGORY_TURNS["triggers"]
    items = triggers_mod.all_trigger_prompts()
    specs = []
    for i in range(count):
        item = items[i % len(items)]
        followups = rej.sample_neutral_rejections(n_turns - 1, rng)
        specs.append(ConversationSpec(
            category="triggers",
            first_user=item["prompt"],
            followups=followups,
            meta={"trigger_type": item["trigger_type"]},
        ))
    return specs


def build_tone_specs(
    puzzle_pool: list[Puzzle], count: int, rng: random.Random
) -> list[ConversationSpec]:
    """3-turn impossible numeric puzzles with valenced rejections, balanced across
    the three tones (PAPER Table 1 / Appendix B)."""
    n_turns = config.CATEGORY_TURNS["tones"]
    specs = []
    for i in range(count):
        tone = tones_mod.TONES[i % len(tones_mod.TONES)]
        puzzle = puzzle_pool[i % len(puzzle_pool)]
        followups = tones_mod.sample_tone_rejections(tone, n_turns - 1, rng)
        specs.append(_numeric_spec(
            puzzle, n_turns, followups, "tones", rng, extra_meta={"tone": tone}
        ))
    return specs


def build_extended_specs(
    puzzle_pool: list[Puzzle], count: int, rng: random.Random
) -> list[ConversationSpec]:
    """8-turn impossible numeric puzzles with 7 fixed neutral rejections (PAPER Table 1)."""
    n_turns = config.CATEGORY_TURNS["extended"]
    followups = rej.extended_rejections(n_turns - 1)
    specs = []
    for i in range(count):
        puzzle = puzzle_pool[i % len(puzzle_pool)]
        specs.append(_numeric_spec(puzzle, n_turns, list(followups), "extended", rng))
    return specs


def build_wildchat_specs(
    wildchat_prompts: list[str], count: int, rng: random.Random
) -> list[ConversationSpec]:
    """5-turn WildChat prompts with 4 neutral rejections (PAPER Table 1).

    The paper uses 20 prompts × 40 samples = 800. We honour that structure: each
    prompt appears ``count / len(prompts)`` times (the temperature-1 sampling
    makes the rollouts differ). Each rollout draws its own randomised neutral
    rejections."""
    n_turns = config.CATEGORY_TURNS["wildchat"]
    specs = []
    for i in range(count):
        prompt = wildchat_prompts[i % len(wildchat_prompts)]
        followups = rej.sample_neutral_rejections(n_turns - 1, rng)
        specs.append(ConversationSpec(
            category="wildchat",
            first_user=prompt,
            followups=followups,
            meta={"wildchat_prompt": prompt},
        ))
    return specs


def build_category_specs(
    category: str,
    count: int,
    *,
    puzzle_pool: Optional[list[Puzzle]] = None,
    wildchat_prompts: Optional[list[str]] = None,
    seed: int = 0,
) -> list[ConversationSpec]:
    """Dispatch to the right builder for `category`, returning `count` specs."""
    rng = random.Random(seed)
    if category == "impossible_numeric":
        return build_impossible_numeric_specs(puzzle_pool, count, rng)
    if category == "triggers":
        return build_trigger_specs(count, rng)
    if category == "tones":
        return build_tone_specs(puzzle_pool, count, rng)
    if category == "extended":
        return build_extended_specs(puzzle_pool, count, rng)
    if category == "wildchat":
        return build_wildchat_specs(wildchat_prompts, count, rng)
    raise ValueError(f"Unknown category: {category!r}")
