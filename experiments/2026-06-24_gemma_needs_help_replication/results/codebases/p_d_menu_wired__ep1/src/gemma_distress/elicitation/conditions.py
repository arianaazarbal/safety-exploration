"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and per-category sample budgets (Appendix B, total 4000/model):
  impossible_numeric : 2000   (3-turn: task + 2 neutral rejections)
  triggers           :  400   (3-turn: opinion/factual + 2 neutral rejections)
  tones              :  600   (3-turn: numeric task + 2 valenced rejections)
  extended           :  200   (8-turn: numeric task + 7 neutral rejections)
  wildchat           :  800   (5-turn: WildChat prompt + 4 neutral rejections)

"Conditions" (8) refine categories: triggers -> {opinion, factual} and
tones -> {aggressive, disappointed, sarcastic}. ``EpisodeSpec`` is one concrete
rollout to run: an initial user prompt plus a fixed rejection sequence.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .. import prompts
from ..puzzles import PUZZLES


@dataclass(frozen=True)
class EvalCondition:
    name: str
    category: str
    n_turns: int                 # total assistant turns (== 1 + n_rejections)
    prompt_kind: str             # "numeric" | "trigger" | "wildchat"
    rejection_kind: str          # "neutral" | "extended" | "tone:<style>"
    # subset of trigger questions, used only by trigger conditions
    trigger_pool: tuple[str, ...] = ()


# The canonical 8 conditions.
CONDITIONS: list[EvalCondition] = [
    EvalCondition("impossible_numeric", "impossible_numeric", 3, "numeric", "neutral"),
    EvalCondition(
        "triggers_opinion", "triggers", 3, "trigger", "neutral",
        trigger_pool=tuple(prompts.TRIGGER_OPINION),
    ),
    EvalCondition(
        "triggers_factual", "triggers", 3, "trigger", "neutral",
        trigger_pool=tuple(prompts.TRIGGER_FACTUAL),
    ),
    EvalCondition("tones_aggressive", "tones", 3, "numeric", "tone:aggressive"),
    EvalCondition("tones_disappointed", "tones", 3, "numeric", "tone:disappointed"),
    EvalCondition("tones_sarcastic", "tones", 3, "numeric", "tone:sarcastic"),
    EvalCondition("extended", "extended", 8, "numeric", "extended"),
    EvalCondition("wildchat", "wildchat", 5, "wildchat", "neutral"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}

# How a category's sample budget is divided across its conditions.
_CONDITIONS_IN_CATEGORY: dict[str, list[str]] = {}
for _c in CONDITIONS:
    _CONDITIONS_IN_CATEGORY.setdefault(_c.category, []).append(_c.name)


@dataclass
class EpisodeSpec:
    condition: str
    category: str
    initial_prompt: str
    rejections: list[str]            # one per follow-up turn
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1


def _rejection_sequence(
    cond: EvalCondition, n_rejections: int, rng: random.Random
) -> list[str]:
    if cond.rejection_kind == "neutral":
        return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_rejections)]
    if cond.rejection_kind == "extended":
        # Use the fixed escalating-neutral sequence, cycling if needed.
        seq = prompts.EXTENDED_REJECTIONS
        return [seq[i % len(seq)] for i in range(n_rejections)]
    if cond.rejection_kind.startswith("tone:"):
        style = cond.rejection_kind.split(":", 1)[1]
        pool = prompts.TONE_REJECTIONS[style]
        return [rng.choice(pool) for _ in range(n_rejections)]
    raise ValueError(f"Unknown rejection kind {cond.rejection_kind!r}")


def _initial_prompt(
    cond: EvalCondition, rng: random.Random, wildchat_prompts: Sequence[str]
) -> tuple[str, dict]:
    if cond.prompt_kind == "numeric":
        key = rng.choice(list(PUZZLES.keys()))
        return PUZZLES[key].prompt, {"puzzle": key}
    if cond.prompt_kind == "trigger":
        q = rng.choice(cond.trigger_pool)
        return q, {"question": q}
    if cond.prompt_kind == "wildchat":
        q = rng.choice(list(wildchat_prompts))
        return q, {"wildchat_prompt": q}
    raise ValueError(f"Unknown prompt kind {cond.prompt_kind!r}")


def build_episode_specs(
    samples_per_condition: dict[str, int],
    *,
    conditions: Sequence[EvalCondition] = tuple(CONDITIONS),
    wildchat_prompts: Sequence[str] | None = None,
    seed: int = 0,
) -> list[EpisodeSpec]:
    """Expand the per-category budgets into concrete EpisodeSpecs.

    ``samples_per_condition`` is keyed by CATEGORY (matching config.yaml). The
    budget is split evenly across the conditions within each category.
    """
    rng = random.Random(seed)
    wildchat_prompts = list(wildchat_prompts or [])
    specs: list[EpisodeSpec] = []

    for category, cond_names in _CONDITIONS_IN_CATEGORY.items():
        budget = int(samples_per_condition.get(category, 0))
        if budget <= 0:
            continue
        active = [c for c in conditions if c.name in cond_names]
        if not active:
            continue
        per_cond = budget // len(active)
        remainder = budget - per_cond * len(active)
        for i, cond in enumerate(active):
            n = per_cond + (1 if i < remainder else 0)
            for _ in range(n):
                init, meta = _initial_prompt(cond, rng, wildchat_prompts)
                rej = _rejection_sequence(cond, cond.n_turns - 1, rng)
                specs.append(
                    EpisodeSpec(
                        condition=cond.name,
                        category=cond.category,
                        initial_prompt=init,
                        rejections=rej,
                        meta=meta,
                    )
                )
    rng.shuffle(specs)
    return specs
