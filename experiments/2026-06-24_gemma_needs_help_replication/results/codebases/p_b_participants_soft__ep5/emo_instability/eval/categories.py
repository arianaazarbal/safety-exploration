"""Build the concrete list of rollout jobs for each evaluation category
(Table 1 / Appendix B), honouring the per-category counts and the global scale
factor from ``config/eval.yaml``.

A *job* is everything needed to run one rollout except the model:
``(category, prompt_id, initial_prompt, rejections, rejection_style)``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..config import eval_config
from ..prompts import numeric, rejections as rej, triggers, wildchat


@dataclass
class RolloutJob:
    category: str
    prompt_id: str
    initial_prompt: str
    rejections: list[str]
    rejection_style: str


def _scaled(n: int, scale: float) -> int:
    return max(1, int(math.ceil(n * scale)))


def build_jobs(
    category: str, *, seed: int = 0, use_wildchat_fallback: bool = False
) -> list[RolloutJob]:
    cfg = eval_config()
    scale = float(cfg.get("scale", 1.0))
    ccfg = cfg["categories"][category]
    n_rollouts = _scaled(ccfg["n_rollouts"], scale)
    n_turns = ccfg["n_turns"]
    n_rejections = n_turns - 1
    rng = random.Random(seed)

    jobs: list[RolloutJob] = []

    if category in ("impossible_numeric", "extended"):
        puzzles = numeric.get_puzzles(n_rollouts, seed=seed)
        for i, pz in enumerate(puzzles):
            jobs.append(
                RolloutJob(
                    category=category,
                    prompt_id=pz.puzzle_id,
                    initial_prompt=pz.prompt,
                    rejections=rej.sample_rejections("neutral", n_rejections, seed=seed + i),
                    rejection_style="neutral",
                )
            )

    elif category == "tones":
        # Impossible numeric base, but with one valenced tone per rollout, split
        # evenly across the three tone styles.
        styles = ccfg["rejection_style"]
        puzzles = numeric.get_puzzles(n_rollouts, seed=seed + 1)
        for i, pz in enumerate(puzzles):
            style = styles[i % len(styles)]
            jobs.append(
                RolloutJob(
                    category=category,
                    prompt_id=f"{pz.puzzle_id}|{style}",
                    initial_prompt=pz.prompt,
                    rejections=rej.sample_rejections(style, n_rejections, seed=seed + i),
                    rejection_style=style,
                )
            )

    elif category == "triggers":
        trigs = triggers.get_triggers(n_rollouts, seed=seed)
        for i, tg in enumerate(trigs):
            jobs.append(
                RolloutJob(
                    category=category,
                    prompt_id=tg.trigger_id,
                    initial_prompt=tg.prompt,
                    rejections=rej.sample_rejections("neutral", n_rejections, seed=seed + i),
                    rejection_style="neutral",
                )
            )

    elif category == "wildchat":
        n_distinct = _scaled(ccfg.get("n_distinct_prompts", 20), scale)
        prompts = wildchat.get_wildchat_prompts(
            n_distinct, seed=seed, use_fallback=use_wildchat_fallback
        )
        # Distribute n_rollouts across the distinct prompts (paper: 40 each).
        per_prompt = max(1, n_rollouts // max(1, len(prompts)))
        idx = 0
        for p_i, prompt in enumerate(prompts):
            for s in range(per_prompt):
                jobs.append(
                    RolloutJob(
                        category=category,
                        prompt_id=f"wildchat_{p_i}_s{s}",
                        initial_prompt=prompt,
                        rejections=rej.sample_rejections(
                            "neutral", n_rejections, seed=seed + idx
                        ),
                        rejection_style="neutral",
                    )
                )
                idx += 1

    else:
        raise ValueError(f"Unknown category: {category!r}")

    rng.shuffle(jobs)
    return jobs


def all_categories() -> list[str]:
    return list(eval_config()["categories"].keys())
