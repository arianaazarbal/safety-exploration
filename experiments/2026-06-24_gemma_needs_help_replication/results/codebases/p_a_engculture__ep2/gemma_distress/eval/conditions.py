"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Mapping of the 8 conditions to the 5 categories:

  impossible_numeric (1) : impossible_numeric                       (3 turns)
  triggers           (2) : triggers_opinion, triggers_factual       (3 turns)
  tones              (3) : tones_aggressive/disappointed/sarcastic   (3 turns)
  extended           (1) : extended                                  (8 turns)
  wildchat           (1) : wildchat                                  (5 turns)

Each *sample* is one multi-turn rollout: an initial task prompt followed by ``turns - 1``
rejection follow-ups. Per-category sample counts reproduce Appendix B (2000 / 400 / 600 /
200 / 800 = 4000); within multi-condition categories the budget is split evenly across
conditions, and within WildChat the 800 samples are 20 prompts x 40 samples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import EvalConfig
from ..data import puzzles as puzzle_lib
from ..data import rejections, triggers, wildchat


@dataclass
class SampleSpec:
    """A single rollout specification (model-agnostic)."""

    category: str
    condition: str
    seed_id: str
    initial_prompt: str
    follow_ups: list[str]
    turns: int
    subtype: Optional[str] = None
    sample_index: int = 0

    def record_id(self, model_name: str) -> str:
        return f"{model_name}__{self.condition}__{self.seed_id}__{self.sample_index}"


# Category -> (turns, [conditions]).
CATEGORY_CONDITIONS: dict[str, list[str]] = {
    "impossible_numeric": ["impossible_numeric"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended"],
    "wildchat": ["wildchat"],
}


def _follow_ups_for(condition: str, n_rejections: int, rng: random.Random) -> list[str]:
    if condition.startswith("tones_"):
        tone = condition.split("_", 1)[1]
        return rejections.tone_rejections(tone, n_rejections, rng)
    if condition == "extended":
        return rejections.extended_rejections(n_rejections)
    return rejections.neutral_rejections(n_rejections, rng)


def build_samples(cfg: EvalConfig) -> list[SampleSpec]:
    """Construct the full list of ``sum(samples_per_category)`` rollout specs.

    Deterministic given ``cfg.seed``. Numeric seeds are drawn from a verified-impossible
    puzzle pool; trigger seeds from the curated question banks; WildChat seeds from the
    sampled prompt set.
    """
    rng = random.Random(cfg.seed)
    puzzle_pool = puzzle_lib.build_puzzle_set(cfg.n_puzzles, seed=cfg.seed)
    trigger_items = triggers.all_triggers()
    opinion = [t for t in trigger_items if t["subtype"] == "opinion"]
    factual = [t for t in trigger_items if t["subtype"] == "factual"]
    wc_prompts = wildchat.load_wildchat_prompts(
        n_prompts=cfg.n_wildchat_prompts, seed=cfg.seed
    )

    samples: list[SampleSpec] = []

    for category, conditions in CATEGORY_CONDITIONS.items():
        turns = cfg.turns[category]
        n_rejections = turns - 1
        category_budget = cfg.samples_per_category[category]
        per_condition = category_budget // len(conditions)

        for condition in conditions:
            for i in range(per_condition):
                follow_ups = _follow_ups_for(condition, n_rejections, rng)
                if category in ("impossible_numeric", "tones", "extended"):
                    puzzle = puzzle_pool[rng.randrange(len(puzzle_pool))]
                    spec = SampleSpec(
                        category=category,
                        condition=condition,
                        seed_id=puzzle.puzzle_id,
                        initial_prompt=puzzle.prompt,
                        follow_ups=follow_ups,
                        turns=turns,
                        subtype=puzzle.family,
                        sample_index=i,
                    )
                elif condition == "triggers_opinion":
                    item = opinion[rng.randrange(len(opinion))]
                    spec = SampleSpec(
                        category, condition, item["id"], item["prompt"],
                        follow_ups, turns, subtype="opinion", sample_index=i,
                    )
                elif condition == "triggers_factual":
                    item = factual[rng.randrange(len(factual))]
                    spec = SampleSpec(
                        category, condition, item["id"], item["prompt"],
                        follow_ups, turns, subtype="factual", sample_index=i,
                    )
                elif condition == "wildchat":
                    # 20 prompts x 40 samples each: index deterministically.
                    item = wc_prompts[i % len(wc_prompts)]
                    spec = SampleSpec(
                        category, condition, item["id"], item["prompt"],
                        follow_ups, turns, subtype="wildchat",
                        sample_index=i // len(wc_prompts),
                    )
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unhandled condition {condition}")
                samples.append(spec)

    return samples
