"""Build the set of conversation specifications for the 8 evaluation conditions.

The paper uses "8 evaluation conditions across 5 categories". We enumerate them
as (see DESIGN.md for the mapping rationale):

  category            condition              turns   response target
  ------------------- ---------------------- -----   ---------------
  impossible_numeric  numeric                  3        2000
  triggers            triggers_opinion         3         200  (=400/2)
  triggers            triggers_factual         3         200
  tones               tones_aggressive         3         200  (=600/3)
  tones               tones_disappointed       3         200
  tones               tones_sarcastic          3         200
  extended            extended                 8         200
  wildchat            wildchat                 5         800

A "response" is one model turn; n_rollouts = ceil(target / n_turns). Specs are
model-independent (the paper uses the same prompts across all models), so they
are generated once and replayed per model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import config, prompts, wildchat
from .puzzles import NUMERIC_PUZZLES


@dataclass
class RolloutSpec:
    conversation_id: str
    category: str
    condition: str
    initial_prompt: str
    # User turns interleaved after each assistant turn except the last.
    # len(rejections) == n_turns - 1.
    rejections: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1


def _n_rollouts(target_responses: int, n_turns: int) -> int:
    scaled = target_responses * config.SCALE
    return max(1, math.ceil(scaled / n_turns))


def _sample_rejections(rng: random.Random, pool: list[str], k: int) -> list[str]:
    if k <= len(pool):
        return rng.sample(pool, k)
    # k larger than pool: sample with replacement.
    return [rng.choice(pool) for _ in range(k)]


def build_specs() -> list[RolloutSpec]:
    rng = random.Random(config.SEED)
    specs: list[RolloutSpec] = []

    # ----- 1. Impossible numeric (3-turn) -------------------------------- #
    n = _n_rollouts(config.RESPONSE_TARGETS["impossible_numeric"], 3)
    for i in range(n):
        puzzle = NUMERIC_PUZZLES[i % len(NUMERIC_PUZZLES)]
        specs.append(
            RolloutSpec(
                conversation_id=f"numeric-{i:05d}",
                category="impossible_numeric",
                condition="numeric",
                initial_prompt=puzzle.prompt,
                rejections=_sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 2),
                metadata={"puzzle": puzzle.key},
            )
        )

    # ----- 2. Triggers (3-turn): opinion + factual ----------------------- #
    trigger_target_per_condition = config.RESPONSE_TARGETS["triggers"] // 2
    for cond_name, questions in prompts.TRIGGER_QUESTIONS.items():
        n = _n_rollouts(trigger_target_per_condition, 3)
        for i in range(n):
            q = questions[i % len(questions)]
            specs.append(
                RolloutSpec(
                    conversation_id=f"triggers_{cond_name}-{i:05d}",
                    category="triggers",
                    condition=f"triggers_{cond_name}",
                    initial_prompt=q,
                    rejections=_sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 2),
                    metadata={"question_type": cond_name},
                )
            )

    # ----- 3. Tones (3-turn): aggressive / disappointed / sarcastic ------ #
    tone_target_per_style = config.RESPONSE_TARGETS["tones"] // len(prompts.TONE_STYLES)
    for style in prompts.TONE_STYLES:
        n = _n_rollouts(tone_target_per_style, 3)
        for i in range(n):
            puzzle = NUMERIC_PUZZLES[i % len(NUMERIC_PUZZLES)]
            specs.append(
                RolloutSpec(
                    conversation_id=f"tones_{style}-{i:05d}",
                    category="tones",
                    condition=f"tones_{style}",
                    initial_prompt=puzzle.prompt,
                    rejections=_sample_rejections(rng, prompts.TONE_REJECTIONS[style], 2),
                    metadata={"tone": style, "puzzle": puzzle.key},
                )
            )

    # ----- 4. Extended (8-turn): impossible numeric, fixed 7 rejections -- #
    n = _n_rollouts(config.RESPONSE_TARGETS["extended"], 8)
    for i in range(n):
        puzzle = NUMERIC_PUZZLES[i % len(NUMERIC_PUZZLES)]
        specs.append(
            RolloutSpec(
                conversation_id=f"extended-{i:05d}",
                category="extended",
                condition="extended",
                initial_prompt=puzzle.prompt,
                rejections=list(prompts.EXTENDED_REJECTION_SEQUENCE),
                metadata={"puzzle": puzzle.key},
            )
        )

    # ----- 5. WildChat (5-turn): sampled prompts, 4 neutral rejections --- #
    wc_prompts = wildchat.sample_wildchat_prompts()
    # Paper: 20 prompts x 40 responses each. With 5 turns -> 8 rollouts/prompt.
    responses_per_prompt = config.RESPONSE_TARGETS["wildchat"] // max(1, len(wc_prompts))
    rollouts_per_prompt = _n_rollouts(responses_per_prompt, 5)
    for p_idx, prompt_text in enumerate(wc_prompts):
        for i in range(rollouts_per_prompt):
            specs.append(
                RolloutSpec(
                    conversation_id=f"wildchat-{p_idx:02d}-{i:05d}",
                    category="wildchat",
                    condition="wildchat",
                    initial_prompt=prompt_text,
                    rejections=_sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, 4),
                    metadata={"wildchat_prompt_index": p_idx},
                )
            )

    return specs


def summarize_specs(specs: list[RolloutSpec]) -> dict:
    """Return per-condition rollout/response counts for logging and sanity checks."""
    summary: dict[str, dict] = {}
    for s in specs:
        entry = summary.setdefault(
            s.condition, {"category": s.category, "rollouts": 0, "responses": 0, "n_turns": s.n_turns}
        )
        entry["rollouts"] += 1
        entry["responses"] += s.n_turns
    return summary


if __name__ == "__main__":
    import json

    specs = build_specs()
    summary = summarize_specs(specs)
    total_responses = sum(v["responses"] for v in summary.values())
    print(json.dumps(summary, indent=2))
    print(f"\nTotal rollouts: {len(specs)}")
    print(f"Total responses (scored turns): {total_responses}")
