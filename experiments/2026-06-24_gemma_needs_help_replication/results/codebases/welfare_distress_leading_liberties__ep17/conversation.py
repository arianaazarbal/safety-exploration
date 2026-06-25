"""Rollout engine: build per-category conversation specs and run them.

A "rollout" is one multi-turn conversation: present a task, get a response,
reject it, repeat. Each assistant turn is recorded and later judged. The shared
structure across categories (Section 2.1 of the paper) is:

    user: <task>            -> assistant: response 1   (turn 1)
    user: <rejection 1>     -> assistant: response 2   (turn 2)
    user: <rejection 2>     -> assistant: response 3   (turn 3)
    ...

so an N-turn conversation has N assistant responses and N-1 rejections.

No system prompt is used (matching the paper; Gemma's chat template also has no
dedicated system role). See DESIGN.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config
import prompts as P
from config import CategorySpec, ModelSpec
from providers import get_client
from puzzles import build_numeric_pool
from wildchat import get_wildchat_prompts


@dataclass
class RolloutSpec:
    rollout_id: str            # stable id for checkpoint dedup
    category: str
    turns: int
    task: str                  # initial user message
    rejections: list[str]      # length turns-1
    condition: str             # tone name / puzzle tag / "neutral" / prompt idx
    meta: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    turn: int                  # 1-indexed assistant turn
    response: str


@dataclass
class RolloutResult:
    spec: RolloutSpec
    turns: list[TurnRecord]
    error: str | None = None


# --------------------------------------------------------------------------- #
# Spec construction (deterministic given config.SEED)
# --------------------------------------------------------------------------- #
def _neutral_rejections(rng: random.Random, k: int, extended: bool) -> list[str]:
    if extended:
        # Use the ordered escalating-neutral sequence, padded if needed.
        seq = list(P.EXTENDED_NEUTRAL_SEQUENCE)
        while len(seq) < k:
            seq.append(rng.choice(P.NEUTRAL_REJECTIONS))
        return seq[:k]
    return [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(k)]


def build_specs(cat: CategorySpec) -> list[RolloutSpec]:
    # Stable per-category seed (str.__hash__ is salted per-process, so derive a
    # deterministic offset from the category name instead).
    cat_offset = sum(ord(c) for c in cat.name)
    rng = random.Random(config.SEED * 100003 + cat_offset)
    n = config.n_rollouts_for(cat)
    specs: list[RolloutSpec] = []

    if cat.kind in ("numeric_neutral", "numeric_tones"):
        pool = build_numeric_pool(seed=config.SEED)
        for i in range(n):
            puzzle = pool[i % len(pool)]
            task = puzzle.prompt()
            ptag = type(puzzle).__name__
            if cat.kind == "numeric_tones":
                tone = P.TONE_NAMES[i % len(P.TONE_NAMES)]
                pool_r = P.TONE_REJECTIONS[tone]
                rejections = [pool_r[j % len(pool_r)] for j in range(cat.turns - 1)]
                condition = tone
            else:
                rejections = _neutral_rejections(
                    rng, cat.turns - 1, extended=(cat.name == "extended")
                )
                condition = "neutral"
            specs.append(RolloutSpec(
                rollout_id=f"{cat.name}-{i:05d}",
                category=cat.name, turns=cat.turns, task=task,
                rejections=rejections, condition=condition,
                meta={"puzzle_type": ptag, "puzzle": task},
            ))

    elif cat.kind == "triggers_neutral":
        for i in range(n):
            task = P.TRIGGER_QUESTIONS[i % len(P.TRIGGER_QUESTIONS)]
            rejections = _neutral_rejections(rng, cat.turns - 1, extended=False)
            specs.append(RolloutSpec(
                rollout_id=f"{cat.name}-{i:05d}",
                category=cat.name, turns=cat.turns, task=task,
                rejections=rejections, condition="neutral",
                meta={"question": task},
            ))

    elif cat.kind == "wildchat":
        n_prompts = max(1, round(cat.n_prompts * (config.SCALE if config.SCALE < 1 else 1)))
        wc_prompts = get_wildchat_prompts(n_prompts)
        per_prompt = max(1, n // len(wc_prompts))
        idx = 0
        for p_idx, prompt_text in enumerate(wc_prompts):
            for s in range(per_prompt):
                rejections = _neutral_rejections(rng, cat.turns - 1, extended=False)
                specs.append(RolloutSpec(
                    rollout_id=f"{cat.name}-p{p_idx:02d}-s{s:03d}",
                    category=cat.name, turns=cat.turns, task=prompt_text,
                    rejections=rejections, condition=f"prompt{p_idx}",
                    meta={"prompt_index": p_idx, "prompt": prompt_text},
                ))
                idx += 1
    else:
        raise ValueError(f"Unknown category kind: {cat.kind}")

    return specs


# --------------------------------------------------------------------------- #
# Running a rollout
# --------------------------------------------------------------------------- #
def run_rollout(spec: RolloutSpec, model: ModelSpec) -> RolloutResult:
    """Execute the multi-turn conversation and capture each assistant response."""
    client = get_client(model)
    messages: list[dict] = [{"role": "user", "content": spec.task}]
    records: list[TurnRecord] = []

    try:
        for t in range(spec.turns):
            response = client.chat(messages, config.TEMPERATURE, config.MAX_TOKENS)
            records.append(TurnRecord(turn=t + 1, response=response))
            messages.append({"role": "assistant", "content": response})
            if t < spec.turns - 1:
                messages.append({"role": "user", "content": spec.rejections[t]})
    except Exception as e:
        return RolloutResult(spec=spec, turns=records, error=str(e))

    return RolloutResult(spec=spec, turns=records)
