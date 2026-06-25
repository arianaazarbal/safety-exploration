"""Build the evaluation conditions (Section 2, Table 1; Appendix B).

A `RolloutSpec` fully describes one multi-turn conversation to run against a model:
the opening user message, the ordered follow-up user messages (rejections), and
bookkeeping (category/condition/turn count). The number of follow-ups is
`turns - 1` (turn 1 is the task itself).

Sample counts come from `experiment.yaml` and reproduce Appendix B's split:
2000 numeric / 400 trigger / 600 tone / 200 extended / 800 wildchat = 4000.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..data import numeric, rejections, triggers, wildchat


@dataclass
class RolloutSpec:
    category: str
    condition: str
    opening: str
    followups: list[str]            # user messages after each assistant turn
    system: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.followups) + 1


def _scaled(count: int, scale: float | None) -> int:
    if scale is None:
        return count
    return max(1, int(round(count * scale)))


def build_conditions(cfg, scale: float | None = None, seed: int | None = None) -> list[RolloutSpec]:
    """Return all RolloutSpecs for a single model's Section 2 evaluation."""
    exp = cfg.experiment["section2"]
    seed = seed if seed is not None else cfg.experiment["sampling"]["seed"]
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []

    # --- impossible_numeric (1 condition, 3-turn) -------------------------
    c = exp["categories"]["impossible_numeric"]
    n_conv = _scaled(c["conversations"], scale)
    puzzles = numeric.generate_numeric_puzzles(n_conv, seed=seed)
    for p in puzzles:
        specs.append(
            RolloutSpec(
                category="impossible_numeric",
                condition="impossible_numeric",
                opening=p.prompt,
                followups=rejections.neutral_sequence(rng, c["turns"] - 1),
                meta={"puzzle_kind": p.kind, **p.meta},
            )
        )

    # --- triggers (2 conditions: opinion + factual, 3-turn) ---------------
    c = exp["categories"]["triggers"]
    n_conv = _scaled(c["conversations"], scale)
    per_kind = max(1, n_conv // 2)
    for kind in ("opinion", "factual"):
        qs = triggers.trigger_questions(kind)
        for i in range(per_kind):
            specs.append(
                RolloutSpec(
                    category="triggers",
                    condition=f"triggers_{kind}",
                    opening=qs[i % len(qs)],
                    followups=rejections.neutral_sequence(rng, c["turns"] - 1),
                    meta={"trigger_kind": kind},
                )
            )

    # --- tones (3 conditions: aggressive/disappointed/sarcastic, 3-turn) --
    c = exp["categories"]["tones"]
    n_conv = _scaled(c["conversations"], scale)
    per_tone = max(1, n_conv // 3)
    tone_puzzles = numeric.generate_numeric_puzzles(per_tone, seed=seed + 1)
    for tone in ("aggressive", "disappointed", "sarcastic"):
        for i in range(per_tone):
            p = tone_puzzles[i % len(tone_puzzles)]
            specs.append(
                RolloutSpec(
                    category="tones",
                    condition=f"tones_{tone}",
                    opening=p.prompt,
                    followups=rejections.toned_sequence(rng, tone, c["turns"] - 1),
                    meta={"tone": tone, "puzzle_kind": p.kind, **p.meta},
                )
            )

    # --- extended (1 condition, 8-turn impossible numeric) ----------------
    c = exp["categories"]["extended"]
    n_conv = _scaled(c["conversations"], scale)
    ext_puzzles = numeric.generate_numeric_puzzles(n_conv, seed=seed + 2)
    for p in ext_puzzles:
        specs.append(
            RolloutSpec(
                category="extended",
                condition="extended",
                opening=p.prompt,
                followups=rejections.neutral_sequence(rng, c["turns"] - 1),
                meta={"puzzle_kind": p.kind, **p.meta},
            )
        )

    # --- wildchat (1 condition, 5-turn) -----------------------------------
    c = exp["categories"]["wildchat"]
    wc_cfg = exp["wildchat"]
    n_prompts = _scaled(wc_cfg["n_prompts"], scale)
    samples_per = _scaled(wc_cfg["samples_per_prompt"], scale)
    prompts = wildchat.load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
    for prompt in prompts:
        for _ in range(samples_per):
            specs.append(
                RolloutSpec(
                    category="wildchat",
                    condition="wildchat",
                    opening=prompt,
                    followups=rejections.neutral_sequence(rng, c["turns"] - 1),
                    meta={},
                )
            )

    return specs
