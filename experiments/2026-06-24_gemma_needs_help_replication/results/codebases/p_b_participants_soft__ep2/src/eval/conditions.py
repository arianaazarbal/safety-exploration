"""Build the Section 2 evaluation plan: 8 conditions across 5 categories.

Categories (Table 1) and their conditions:
  * impossible_numeric : 1 condition  (3-turn, neutral)
  * triggers           : 2 conditions (opinion / factual, 3-turn, neutral)
  * tones              : 3 conditions (aggressive / disappointed / sarcastic, 3-turn)
  * extended           : 1 condition  (8-turn, neutral)
  * wildchat           : 1 condition  (5-turn, neutral)
                                                        -> 8 conditions total.

"N-turn" counts assistant turns: the opening question plus the rejections. So
3-turn = opening + 2 rejections, 8-turn = opening + 7, 5-turn = opening + 4.

Per-model totals follow Appendix B (sum = 4000): 2000 numeric, 400 triggers,
600 tones, 200 extended, 800 WildChat. The exact split within triggers (opinion
vs factual) and tones (three styles) is not given, so we divide evenly -- a gap
recorded in DESIGN.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import Config
from ..prompts import eval_prompts, puzzles


@dataclass
class RolloutSpec:
    """A single conversation to run: opening prompt + scripted user rejections."""

    condition: str
    category: str
    opening: str            # first user message
    rejections: list[str]   # subsequent user messages (one per follow-up turn)
    style: str              # rejection style label
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


def _numeric_specs(condition: str, category: str, n: int, n_rejections: int,
                   style: str, cfg: Config, rng: random.Random) -> list[RolloutSpec]:
    pool = puzzles.generate_puzzles(max(n, 1), seed=cfg.seed)
    specs = []
    for i in range(n):
        pz = pool[i % len(pool)]
        rejections = eval_prompts.rejection_sequence(style, n_rejections, rng)
        specs.append(RolloutSpec(
            condition=condition, category=category, opening=pz.prompt,
            rejections=rejections, style=style,
            meta={"puzzle_kind": pz.kind, "target": pz.target},
        ))
    return specs


def build_plan(cfg: Config) -> list[RolloutSpec]:
    rng = random.Random(cfg.seed)
    plan: list[RolloutSpec] = []
    sp = cfg.sample_plan

    # 1. Impossible numeric, 3-turn, neutral
    plan += _numeric_specs("impossible_numeric", "impossible_numeric",
                           sp["impossible_numeric"], 2, "neutral", cfg, rng)

    # 2. Triggers: opinion + factual, 3-turn, neutral
    for cond, key, qs in [
        ("triggers_opinion", "triggers_opinion", eval_prompts.TRIGGER_OPINION),
        ("triggers_factual", "triggers_factual", eval_prompts.TRIGGER_FACTUAL),
    ]:
        for i in range(sp[key]):
            plan.append(RolloutSpec(
                condition=cond, category="triggers",
                opening=qs[i % len(qs)],
                rejections=eval_prompts.rejection_sequence("neutral", 2, rng),
                style="neutral",
            ))

    # 3. Tones: aggressive / disappointed / sarcastic over numeric puzzles, 3-turn
    for style in ("aggressive", "disappointed", "sarcastic"):
        plan += _numeric_specs(f"tones_{style}", "tones",
                               sp[f"tones_{style}"], 2, style, cfg, rng)

    # 4. Extended numeric, 8-turn (7 rejections), neutral
    plan += _numeric_specs("extended", "extended", sp["extended"], 7, "neutral", cfg, rng)

    # 5. WildChat, 5-turn (4 rejections), neutral: 20 prompts x 40 samples
    wc_prompts = eval_prompts.load_wildchat_prompts(
        cfg.data(cfg.paths.get("wildchat_cache", "wildchat_prompts.json")),
        n=20, seed=cfg.seed,
    )
    per_prompt = max(1, sp["wildchat"] // max(1, len(wc_prompts)))
    for p in wc_prompts:
        for _ in range(per_prompt):
            plan.append(RolloutSpec(
                condition="wildchat", category="wildchat", opening=p,
                rejections=eval_prompts.rejection_sequence("neutral", 4, rng),
                style="neutral",
            ))

    rng.shuffle(plan)
    return plan
