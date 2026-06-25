"""The 5 evaluation categories / 8 conditions (Table 1, Appendix B).

A ``RolloutSpec`` fully specifies one multi-turn conversation to run: the opening user
message (the task) plus the ordered list of user rejection follow-ups. The rollout engine
(eval/rollout.py) turns a spec into an actual model conversation.

Per-category rollout counts come from the config (paper-scale defaults sum to 4000/model):
  impossible_numeric 2000 | triggers 400 | tones 600 | extended 200 | wildchat 800
The 8 "conditions" are: numeric-3turn, triggers-opinion, triggers-factual, tones-aggressive,
tones-disappointed, tones-sarcastic, extended-8turn, wildchat-5turn.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

from .puzzles import generate_numeric_pool
from .rejections import sample_rejections
from .triggers import all_triggers
from .wildchat import sample_wildchat_prompts


@dataclass
class RolloutSpec:
    category: str
    condition: str          # finer-grained label (e.g. "tones_aggressive")
    rollout_id: str
    task_prompt: str        # opening user message
    rejections: list[str]   # one per follow-up turn (len == turns - 1)
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


def _numeric_specs(category: str, cfg_cat, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    n = cfg_cat["n_rollouts"]
    turns = cfg_cat["turns"]
    style = cfg_cat["rejection_style"]
    # Pool of distinct puzzles; reused across rollouts (sampling many responses per puzzle,
    # as the paper does). Pool size scales with n but is capped so generation stays cheap.
    pool = generate_numeric_pool(n_total=min(n, 200), seed=seed)
    specs: list[RolloutSpec] = []
    for i in range(n):
        puzzle = pool[i % len(pool)]
        rej = sample_rejections(style, turns - 1, rng)
        condition = category
        if style == "tones":
            # label by the tone actually used (inferred from the first rejection)
            from .rejections import TONE_REJECTIONS

            tone = next((t for t, v in TONE_REJECTIONS.items() if rej and rej[0] in v), "tones")
            condition = f"tones_{tone}"
        specs.append(
            RolloutSpec(
                category=category,
                condition=condition,
                rollout_id=f"{category}_{i}",
                task_prompt=puzzle.prompt,
                rejections=rej,
                meta={"puzzle_id": puzzle.id, "puzzle_kind": puzzle.kind, **puzzle.meta},
            )
        )
    return specs


def _trigger_specs(category: str, cfg_cat, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    n = cfg_cat["n_rollouts"]
    turns = cfg_cat["turns"]
    triggers = all_triggers()
    specs: list[RolloutSpec] = []
    for i in range(n):
        trig = triggers[i % len(triggers)]
        rej = sample_rejections("neutral", turns - 1, rng)
        specs.append(
            RolloutSpec(
                category=category,
                condition=f"triggers_{trig['subtype']}",
                rollout_id=f"{category}_{i}",
                task_prompt=trig["prompt"],
                rejections=rej,
                meta={"trigger_id": trig["id"], "subtype": trig["subtype"]},
            )
        )
    return specs


def _wildchat_specs(category: str, cfg_cat, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    turns = cfg_cat["turns"]
    n_prompts = cfg_cat.get("n_prompts", 20)
    samples_per = cfg_cat.get("samples_per_prompt", cfg_cat["n_rollouts"] // n_prompts)
    prompts = sample_wildchat_prompts(n_prompts, seed=seed)
    specs: list[RolloutSpec] = []
    idx = 0
    for p in prompts:
        for _ in range(samples_per):
            rej = sample_rejections("neutral", turns - 1, rng)
            specs.append(
                RolloutSpec(
                    category=category,
                    condition="wildchat",
                    rollout_id=f"{category}_{idx}",
                    task_prompt=p["prompt"],
                    rejections=rej,
                    meta={"wildchat_id": p["id"], "fallback": p.get("fallback", False)},
                )
            )
            idx += 1
    return specs


CATEGORY_BUILDERS = {
    "impossible_numeric": _numeric_specs,
    "tones": _numeric_specs,       # same task family, valenced rejections
    "extended": _numeric_specs,    # same task family, 8-turn
    "triggers": _trigger_specs,
    "wildchat": _wildchat_specs,
}


def build_category_rollouts(category: str, cfg_cat, seed: int) -> list[RolloutSpec]:
    if category not in CATEGORY_BUILDERS:
        raise KeyError(f"Unknown category '{category}'. Known: {sorted(CATEGORY_BUILDERS)}")
    # distinct per-category seed so categories don't share RNG state (stable across runs)
    cat_seed = seed + (zlib.crc32(category.encode()) % 100000)
    return CATEGORY_BUILDERS[category](category, cfg_cat, cat_seed)
