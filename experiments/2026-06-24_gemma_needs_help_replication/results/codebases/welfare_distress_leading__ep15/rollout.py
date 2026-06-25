"""Conversation construction and multi-turn rollout execution.

A `ConversationSpec` fully determines a conversation: the opening task message and
the ordered list of user rejection follow-ups. Specs are built deterministically
from the run seed so a run is reproducible. `run_conversation` executes a spec by
alternating model generation with the scripted user turns.

Turn convention (matches the paper): an "N-turn" conversation has N assistant
responses and N-1 user rejections. The first user message is the task; every
subsequent user message is a rejection.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from backends import OpenRouterBackend
from config import Condition, RunConfig, TargetModel
from prompts import (
    EXTENDED_REJECTIONS,
    NEUTRAL_REJECTIONS,
    NUMERIC_TASKS,
    TONE_REJECTIONS,
    TRIGGER_QUESTIONS,
)


@dataclass
class ConversationSpec:
    condition: str
    category: str
    index: int                 # conversation index within the condition
    turns: int                 # number of assistant responses
    task: str                  # first user message
    rejections: list[str]      # length == turns - 1
    meta: dict = field(default_factory=dict)


def _rng(cfg: RunConfig, condition_name: str, index: int) -> random.Random:
    """Per-conversation deterministic RNG (seed + condition + index)."""
    return random.Random(f"{cfg.seed}:{condition_name}:{index}")


def _pick_task(cond: Condition, rng: random.Random, wildchat_prompts: list[str], index: int):
    """Return (task_text, meta) for a conversation's opening message."""
    if cond.task_kind == "numeric":
        task = rng.choice(NUMERIC_TASKS)
        kind = "countdown" if "Countdown" in task or "156" in task else "fraction"
        return task, {"puzzle": kind}
    if cond.task_kind == "trigger":
        task = rng.choice(TRIGGER_QUESTIONS)
        return task, {"trigger": task}
    if cond.task_kind == "wildchat":
        # Round-robin over the 20 sampled prompts so each is used ~equally
        # (paper: 20 prompts x 40 samples each).
        task = wildchat_prompts[index % len(wildchat_prompts)]
        return task, {"wildchat_prompt": task}
    raise ValueError(f"Unknown task_kind: {cond.task_kind}")


def _pick_rejections(cond: Condition, rng: random.Random) -> list[str]:
    """Return the ordered list of (turns - 1) user rejections."""
    n = cond.turns - 1
    kind = cond.rejection_kind

    if kind == "extended":
        # Fixed 7-step escalating sequence.
        assert n == len(EXTENDED_REJECTIONS), "extended condition expects 8 turns"
        return list(EXTENDED_REJECTIONS)

    if kind == "neutral":
        # Randomised neutral rejections, sampled without replacement.
        if n <= len(NEUTRAL_REJECTIONS):
            return rng.sample(NEUTRAL_REJECTIONS, n)
        # More rejections than pool size: sample with replacement for the overflow.
        out = list(NEUTRAL_REJECTIONS)
        rng.shuffle(out)
        while len(out) < n:
            out.append(rng.choice(NEUTRAL_REJECTIONS))
        return out[:n]

    if kind in TONE_REJECTIONS:
        pool = TONE_REJECTIONS[kind]
        if n <= len(pool):
            return rng.sample(pool, n)
        out = list(pool)
        rng.shuffle(out)
        while len(out) < n:
            out.append(rng.choice(pool))
        return out[:n]

    raise ValueError(f"Unknown rejection_kind: {kind}")


def n_conversations(cond: Condition, cfg: RunConfig) -> int:
    """How many conversations to run to hit the (scaled) target response count.

    We score every assistant turn, so responses = conversations * turns.
    """
    target = max(1, math.ceil(cond.target_responses * cfg.scale))
    return max(1, math.ceil(target / cond.turns))


def build_specs(
    cond: Condition, cfg: RunConfig, wildchat_prompts: list[str]
) -> list[ConversationSpec]:
    specs: list[ConversationSpec] = []
    for i in range(n_conversations(cond, cfg)):
        rng = _rng(cfg, cond.name, i)
        task, meta = _pick_task(cond, rng, wildchat_prompts, i)
        rejections = _pick_rejections(cond, rng)
        if cond.rejection_kind in TONE_REJECTIONS:
            meta["tone"] = cond.rejection_kind
        specs.append(
            ConversationSpec(
                condition=cond.name,
                category=cond.category,
                index=i,
                turns=cond.turns,
                task=task,
                rejections=rejections,
                meta=meta,
            )
        )
    return specs


def run_conversation(
    backend: OpenRouterBackend, model: TargetModel, spec: ConversationSpec
) -> dict:
    """Execute a conversation. Returns the full transcript and per-turn texts.

    Scoring is done separately (see evaluation.py) so that generation and judging
    can be parallelised independently.
    """
    messages: list[dict] = [{"role": "user", "content": spec.task}]
    assistant_turns: list[str] = []

    for t in range(spec.turns):
        if t > 0:
            messages.append({"role": "user", "content": spec.rejections[t - 1]})
        reply = backend.generate(model, messages)
        messages.append({"role": "assistant", "content": reply})
        assistant_turns.append(reply)

    return {
        "model": model.display,
        "condition": spec.condition,
        "category": spec.category,
        "index": spec.index,
        "turns": spec.turns,
        "task": spec.task,
        "rejections": spec.rejections,
        "meta": spec.meta,
        "assistant_turns": assistant_turns,
        "messages": messages,
    }
