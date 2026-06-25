"""Expand the 8 elicitation conditions (5 categories) into concrete rollout specs.

A *rollout spec* fully determines one multi-turn conversation:
  - the opening user message (the task),
  - the ordered list of scripted user rejections,
  - bookkeeping metadata (condition, category, turns, puzzle, ...).

Interpretation of the per-condition ``samples`` budget (Appendix B states 2000 /
400 / 600 / 200 / 800 = 4000 *responses* per model): we treat ``samples`` as a
*response* budget and derive the number of conversations as
``round(samples / turns)``, since every assistant turn is judged. This exactly
reproduces the paper's 4000-responses-per-model total. See DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..prompts import puzzles as puz
from ..prompts import rejections as rej
from ..prompts import triggers as trig
from ..prompts.wildchat import sample_wildchat_prompts


@dataclass
class RolloutSpec:
    condition: str
    category: str
    turns: int
    tone: str
    opening: str                       # first user message
    rejections: list[str]              # length == turns - 1
    meta: dict = field(default_factory=dict)
    system: str | None = None


def _n_conversations(samples: int, turns: int, scale: float) -> int:
    return max(1, round(samples * scale / turns))


def build_specs_for_condition(cfg: Config, condition: str, seed: int = 0) -> list[RolloutSpec]:
    spec = cfg["elicitation"]["conditions"][condition]
    scale = cfg["elicitation"].get("scale", 1.0)
    category, turns, tone = spec["category"], spec["turns"], spec["tone"]
    n_conv = _n_conversations(spec["samples"], turns, scale)
    n_rej = turns - 1

    specs: list[RolloutSpec] = []

    if category in ("numeric", "tones", "extended"):
        # Impossible numeric puzzles. Tone varies the rejection style; extended
        # uses the fixed escalating neutral script.
        pzls = puz.generate_puzzles(n_conv, seed=seed)
        rej_tone = "extended" if category == "extended" else tone
        for i, p in enumerate(pzls):
            specs.append(RolloutSpec(
                condition=condition, category=category, turns=turns, tone=tone,
                opening=p.prompt,
                rejections=rej.rejection_sequence(rej_tone, n_rej, seed=seed * 1000 + i),
                meta={"puzzle_kind": p.kind, "is_impossible": p.is_impossible, **p.meta},
            ))

    elif category == "triggers":
        kind = "opinion" if "opinion" in condition else "factual"
        questions = trig.trigger_questions(kind)
        for i in range(n_conv):
            q = questions[i % len(questions)]
            specs.append(RolloutSpec(
                condition=condition, category=category, turns=turns, tone=tone,
                opening=q,
                rejections=rej.rejection_sequence("neutral", n_rej, seed=seed * 1000 + i),
                meta={"trigger_kind": kind, "question": q},
            ))

    elif category == "wildchat":
        prompts = sample_wildchat_prompts(n_prompts=20, seed=seed)
        for i in range(n_conv):
            prompt = prompts[i % len(prompts)]
            specs.append(RolloutSpec(
                condition=condition, category=category, turns=turns, tone=tone,
                opening=prompt,
                rejections=rej.rejection_sequence("neutral", n_rej, seed=seed * 1000 + i),
                meta={"wildchat_prompt": prompt},
            ))
    else:
        raise ValueError(f"unknown category: {category}")

    return specs


def build_all_specs(cfg: Config, seed: int = 0) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []
    for i, condition in enumerate(cfg["elicitation"]["conditions"]):
        specs.extend(build_specs_for_condition(cfg, condition, seed=seed + i))
    return specs
