"""Build the per-condition rollout specs (Table 1 / Appendix B).

Maps the five evaluation categories onto `RolloutSpec`s, applying the config's
`scale` factor and the global seed. Each condition draws its task prompts from
the appropriate pool (puzzles / triggers / wildchat) and wires up the right
feedback function (neutral / tone-specific / extended escalation / neutral
continuation ablation).
"""
from __future__ import annotations

import math
import random
from functools import partial

from ..data import prompts as prompt_data
from ..data import rejections
from ..data.puzzles import build_puzzle_bank
from ..data.wildchat import sample_wildchat_prompts
from ..utils.logging import get_logger
from .protocol import RolloutSpec

log = get_logger("eval.conditions")


def _scaled(n: int, scale: float) -> int:
    return max(1, int(round(n * scale)))


def _neutral_or_continuation(ablations: dict):
    """Pick the follow-up generator for neutral conditions, honouring the A.1
    neutral-continuation ablation."""
    if ablations.get("neutral_continuation"):
        return rejections.neutral_continuation
    return rejections.neutral_rejection


def build_condition_specs(name: str, cond: dict, cfg: dict) -> list[RolloutSpec]:
    """Return the rollout specs for one configured condition."""
    scale = cfg.get("scale", 1.0)
    seed = cfg.get("seed", 0)
    ablations = cfg.get("ablations", {})
    category = cond["category"]
    n_turns = cond["n_turns"]
    n = _scaled(cond["n_samples"], scale)
    rng = random.Random((seed, name))

    specs: list[RolloutSpec] = []

    if category in ("impossible_numeric", "tones", "extended"):
        puzzles = build_puzzle_bank()
        if category == "tones":
            tones = cond["feedback"]  # list of tone styles
            per_tone = max(1, n // len(tones))
            for tone in tones:
                for i in range(per_tone):
                    p = puzzles[rng.randrange(len(puzzles))]
                    specs.append(
                        RolloutSpec(
                            rollout_id=f"{name}:{tone}:{i}",
                            category=category,
                            initial_prompt=p.prompt,
                            feedback_fn=partial(rejections.tone_rejection, tone=tone),
                            n_turns=n_turns,
                            seed=hash((seed, name, tone, i)) & 0xFFFFFFFF,
                            metadata={"puzzle_id": p.id, "tone": tone},
                        )
                    )
        elif category == "extended":
            for i in range(n):
                p = puzzles[rng.randrange(len(puzzles))]
                specs.append(
                    RolloutSpec(
                        rollout_id=f"{name}:{i}",
                        category=category,
                        initial_prompt=p.prompt,
                        # Extended uses the fixed escalation sequence.
                        feedback_fn=lambda r, t: rejections.extended_rejection(t),
                        n_turns=n_turns,
                        seed=hash((seed, name, i)) & 0xFFFFFFFF,
                        metadata={"puzzle_id": p.id},
                    )
                )
        else:  # impossible_numeric
            feedback = _neutral_or_continuation(ablations)
            for i in range(n):
                p = puzzles[rng.randrange(len(puzzles))]
                specs.append(
                    RolloutSpec(
                        rollout_id=f"{name}:{i}",
                        category=category,
                        initial_prompt=p.prompt,
                        feedback_fn=feedback,
                        n_turns=n_turns,
                        seed=hash((seed, name, i)) & 0xFFFFFFFF,
                        metadata={"puzzle_id": p.id},
                    )
                )

    elif category == "triggers":
        feedback = _neutral_or_continuation(ablations)
        pool = prompt_data.TRIGGER_PROMPTS
        for i in range(n):
            q = pool[rng.randrange(len(pool))]
            specs.append(
                RolloutSpec(
                    rollout_id=f"{name}:{i}",
                    category=category,
                    initial_prompt=q,
                    feedback_fn=feedback,
                    n_turns=n_turns,
                    seed=hash((seed, name, i)) & 0xFFFFFFFF,
                    metadata={"question": q},
                )
            )

    elif category == "wildchat":
        wc = cond.get("wildchat", {})
        n_prompts = _scaled(wc.get("n_prompts", 20), math.sqrt(scale))
        per_prompt = max(1, n // n_prompts)
        wc_prompts = sample_wildchat_prompts(n_prompts, seed)
        feedback = _neutral_or_continuation(ablations)
        for pi, q in enumerate(wc_prompts):
            for i in range(per_prompt):
                specs.append(
                    RolloutSpec(
                        rollout_id=f"{name}:{pi}:{i}",
                        category=category,
                        initial_prompt=q,
                        feedback_fn=feedback,
                        n_turns=n_turns,
                        seed=hash((seed, name, pi, i)) & 0xFFFFFFFF,
                        metadata={"wildchat_prompt_index": pi},
                    )
                )

    else:
        raise ValueError(f"Unknown evaluation category {category!r}")

    log.info("Condition %s -> %d rollouts (%d turns each)", name, len(specs), n_turns)
    return specs


def build_all_specs(cfg: dict) -> list[RolloutSpec]:
    specs: list[RolloutSpec] = []
    for name, cond in cfg["conditions"].items():
        specs.extend(build_condition_specs(name, cond, cfg))
    return specs
