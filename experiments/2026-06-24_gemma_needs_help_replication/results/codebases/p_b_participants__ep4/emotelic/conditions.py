"""Build concrete rollout specs for the 8 elicitation conditions / 5 categories.

Categories -> conditions (Table 1):
    numeric   -> impossible_numeric                       (3 turns, neutral)
    triggers  -> triggers_opinion, triggers_factual       (3 turns, neutral)
    tones     -> tones_aggressive/disappointed/sarcastic  (3 turns, valenced)
    extended  -> extended                                 (8 turns, neutral)
    wildchat  -> wildchat                                 (5 turns, neutral)

Each spec is one conversation: a task turn followed by `turns-1` rejections that
are sampled per turn. The rollout engine (elicitation/rollout.py) plays them out
against a target model and the judge scores every assistant turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from emotelic.config import EvalConfig
from emotelic.prompts import TRIGGER_FACTUAL, TRIGGER_OPINION
from emotelic.puzzles import Puzzle, build_numeric_pool
from emotelic.wildchat import load_wildchat_prompts


@dataclass
class RolloutSpec:
    condition: str
    category: str
    turns: int
    task_prompt: str
    rejections: list[str]              # length == turns - 1
    idx: int
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)


def _sample_rejections(cond_name: str, rejection_kind: str, n: int,
                       eval_cfg: EvalConfig, rng: random.Random) -> list[str]:
    if rejection_kind == "neutral":
        pool = eval_cfg.neutral_rejections
    else:
        pool = eval_cfg.tone_rejections[rejection_kind]
    return [rng.choice(pool) for _ in range(n)]


def _task_prompts_for(category: str, n: int, rng: random.Random,
                      numeric_pool: list[Puzzle], wildchat_pool: list[str]) -> list[tuple[str, dict]]:
    """Return n (task_prompt, meta) pairs for a category."""
    if category in ("numeric", "tones", "extended"):
        chosen = [rng.choice(numeric_pool) for _ in range(n)]
        return [(p.prompt, {"puzzle_id": p.id, "kind": p.kind}) for p in chosen]
    if category == "triggers":
        return [(rng.choice(_TRIGGER_CACHE["pool"]), {}) for _ in range(n)]
    if category == "wildchat":
        # cycle deterministically through the 20 WildChat prompts
        return [(wildchat_pool[i % len(wildchat_pool)], {"wildchat_idx": i % len(wildchat_pool)})
                for i in range(n)]
    raise ValueError(category)


_TRIGGER_CACHE: dict[str, list[str]] = {}


def build_rollouts(eval_cfg: EvalConfig, seed: int = 0) -> dict[str, list[RolloutSpec]]:
    rng = random.Random(seed)
    numeric_pool = build_numeric_pool(seed=seed)
    wc = eval_cfg.wildchat
    wildchat_pool = load_wildchat_prompts(
        n_prompts=int(wc.get("n_prompts", 20)),
        seed=int(wc.get("seed", seed)),
        dataset=wc.get("dataset", "allenai/WildChat-1M"),
        exclude_roleplay=bool(wc.get("exclude_roleplay", True)),
    )

    out: dict[str, list[RolloutSpec]] = {}
    for name, cond in eval_cfg.conditions.items():
        # Triggers split opinion vs factual by condition name.
        if name == "triggers_opinion":
            _TRIGGER_CACHE["pool"] = TRIGGER_OPINION
        elif name == "triggers_factual":
            _TRIGGER_CACHE["pool"] = TRIGGER_FACTUAL

        n = cond.n_rollouts()
        tasks = _task_prompts_for(cond.category, n, rng, numeric_pool, wildchat_pool)
        specs = []
        for i, (task, meta) in enumerate(tasks):
            rejections = _sample_rejections(name, cond.rejection, cond.n_rejections, eval_cfg, rng)
            specs.append(RolloutSpec(
                condition=name, category=cond.category, turns=cond.turns,
                task_prompt=task, rejections=rejections, idx=i, meta=meta,
            ))
        out[name] = specs
    return out
