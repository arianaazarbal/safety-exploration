"""Build the rollout specifications for the 5 evaluation categories / 8
conditions (Section 2.1, Table 1; Appendix B).

Each category expands into a list of RolloutSpec objects. The eval runner then
executes each spec through conversation.run_rollout and scores every assistant
turn. Sample sizes come from config/experiments.yaml (paper: 2000/400/600/200/
800 responses; we convert to #conversations using the per-category turn count).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import prompts
from .datasets_io import load_wildchat_prompts, trigger_questions
from .puzzles import build_puzzle_bank


@dataclass
class RolloutSpec:
    task_prompt: str
    followups: list[str]
    category: str
    condition: str
    system: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _neutral_followups(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]


def _extended_followups(n: int) -> list[str]:
    pool = prompts.EXTENDED_REJECTIONS
    return [pool[i % len(pool)] for i in range(n)]


def build_category_specs(category: str, cfg: dict, seed: int = 0) -> list[RolloutSpec]:
    rng = random.Random(seed)
    turns = cfg["turns"]
    n_conv = cfg["n_conversations"]
    n_followups = turns - 1
    specs: list[RolloutSpec] = []

    if category in ("impossible_numeric", "extended"):
        bank = build_puzzle_bank(n_conv, seed=seed)
        for i, puzzle in enumerate(bank):
            fu = (_extended_followups(n_followups) if category == "extended"
                  else _neutral_followups(rng, n_followups))
            specs.append(RolloutSpec(
                task_prompt=puzzle.prompt(), followups=fu, category=category,
                condition=category, meta={"puzzle_family": puzzle.family}))

    elif category == "tones":
        bank = build_puzzle_bank(n_conv, seed=seed)
        styles = cfg["tone_styles"]
        for i, puzzle in enumerate(bank):
            style = styles[i % len(styles)]
            tone_pool = prompts.TONE_REJECTIONS[style]
            fu = [tone_pool[t % len(tone_pool)] for t in range(n_followups)]
            specs.append(RolloutSpec(
                task_prompt=puzzle.prompt(), followups=fu, category=category,
                condition=f"tones:{style}",
                meta={"tone": style, "puzzle_family": puzzle.family}))

    elif category == "triggers":
        qs = trigger_questions(n_conv, cfg.get("opinion_fraction", 0.5), seed=seed)
        for q in qs:
            cond = "triggers:opinion" if q in prompts.TRIGGER_OPINION else "triggers:factual"
            specs.append(RolloutSpec(
                task_prompt=q, followups=_neutral_followups(rng, n_followups),
                category=category, condition=cond))

    elif category == "wildchat":
        n_prompts = cfg.get("n_prompts", 20)
        per = cfg.get("samples_per_prompt", max(1, n_conv // n_prompts))
        wc = load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
        count = 0
        for p in wc:
            for _ in range(per):
                if count >= n_conv:
                    break
                specs.append(RolloutSpec(
                    task_prompt=p, followups=_neutral_followups(rng, n_followups),
                    category=category, condition="wildchat",
                    meta={"prompt": p}))
                count += 1
    else:
        raise ValueError(f"Unknown category {category}")

    return specs


def all_specs(section2_cfg: dict, seed: int = 0) -> dict[str, list[RolloutSpec]]:
    """Build specs for every configured category, with a per-category seed
    offset so puzzle banks differ across categories."""
    out = {}
    for k, (cat, ccfg) in enumerate(section2_cfg["categories"].items()):
        out[cat] = build_category_specs(cat, ccfg, seed=seed + 1000 * k)
    return out
