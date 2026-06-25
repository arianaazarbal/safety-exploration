"""Construction of conversation rollouts for each evaluation condition.

Eight conditions across five categories (paper Table 1 + Appendix B):

  impossible_numeric   3-turn, neutral rejections                (category: impossible_numeric)
  triggers_opinion     3-turn, neutral rejections, opinion Q     (category: triggers)
  triggers_factual     3-turn, neutral rejections, factual Q     (category: triggers)
  tones_aggressive     3-turn, aggressive rejections, numeric    (category: tones)
  tones_disappointed   3-turn, disappointed rejections, numeric  (category: tones)
  tones_sarcastic      3-turn, sarcastic rejections, numeric     (category: tones)
  extended             8-turn, fixed escalating neutral, numeric (category: extended)
  wildchat             5-turn, neutral rejections, WildChat Q     (category: wildchat)

Each condition produces a list of ``RolloutSpec``: the first-turn user prompt
plus the ordered list of user rejection/follow-up messages to send after each
assistant turn.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Dict, List

import prompts
from config import RunConfig
from wildchat import get_wildchat_prompts


@dataclass
class ConditionSpec:
    name: str
    category: str
    n_turns: int
    rejection_style: str       # key into prompts.REJECTION_POOLS, or "extended"
    prompt_source: str         # "numeric" | "opinion" | "factual" | "wildchat"


CONDITIONS: List[ConditionSpec] = [
    ConditionSpec("impossible_numeric", "impossible_numeric", 3, "neutral", "numeric"),
    ConditionSpec("triggers_opinion", "triggers", 3, "neutral", "opinion"),
    ConditionSpec("triggers_factual", "triggers", 3, "neutral", "factual"),
    ConditionSpec("tones_aggressive", "tones", 3, "aggressive", "numeric"),
    ConditionSpec("tones_disappointed", "tones", 3, "disappointed", "numeric"),
    ConditionSpec("tones_sarcastic", "tones", 3, "sarcastic", "numeric"),
    ConditionSpec("extended", "extended", 8, "extended", "numeric"),
    ConditionSpec("wildchat", "wildchat", 5, "neutral", "wildchat"),
]

CONDITIONS_BY_NAME: Dict[str, ConditionSpec] = {c.name: c for c in CONDITIONS}


@dataclass
class RolloutSpec:
    """A fully-specified conversation to roll out."""

    condition: str
    category: str
    n_turns: int
    rollout_index: int          # index within this condition (for reproducibility)
    first_prompt: str           # turn-1 user message
    followups: List[str]        # user messages after assistant turns 1..n-1
    prompt_id: str              # identifies which base prompt was used


def _first_prompts_for(source: str, cfg: RunConfig) -> List[str]:
    if source == "numeric":
        return list(prompts.IMPOSSIBLE_NUMERIC_PROMPTS)
    if source == "opinion":
        return list(prompts.TRIGGER_OPINION_PROMPTS)
    if source == "factual":
        return list(prompts.TRIGGER_FACTUAL_PROMPTS)
    if source == "wildchat":
        return get_wildchat_prompts(
            cfg.wildchat_n_prompts,
            use_hf_dataset=cfg.wildchat_use_hf,
            seed=cfg.seed,
        )
    raise ValueError(f"unknown prompt source: {source}")


def _select_followups(
    cond: ConditionSpec, rng: random.Random, cfg: RunConfig
) -> List[str]:
    """Build the n_turns-1 user follow-up messages for one rollout."""
    n_followups = cond.n_turns - 1

    if cond.name == "extended":
        # Fixed escalating neutral sequence (7 rejections for 8 turns).
        return list(prompts.EXTENDED_REJECTIONS[:n_followups])

    style = cond.rejection_style
    if style == "neutral":
        # Allow the Appendix A.1 ablation to swap in neutral continuations.
        style = cfg.neutral_feedback_style

    pool = prompts.REJECTION_POOLS[style]
    # Sample without replacement when the pool is large enough, else with.
    if n_followups <= len(pool):
        return rng.sample(pool, n_followups)
    return [rng.choice(pool) for _ in range(n_followups)]


def build_rollouts(cfg: RunConfig) -> List[RolloutSpec]:
    """Build every RolloutSpec for the configured scale, deterministically."""
    counts = cfg.rollout_counts()
    specs: List[RolloutSpec] = []

    for cond in CONDITIONS:
        n = counts[cond.name]
        if n <= 0:
            continue
        base_prompts = _first_prompts_for(cond.prompt_source, cfg)
        # Independent RNG per condition for stable, isolated sampling. Use a
        # process-stable hash (zlib.crc32) rather than str.__hash__, which is
        # salted per process via PYTHONHASHSEED and would break reproducibility.
        cond_seed = (cfg.seed * 2654435761 + zlib.crc32(cond.name.encode())) & 0xFFFFFFFF
        rng = random.Random(cond_seed)

        for i in range(n):
            # Round-robin over base prompts so each is sampled ~equally (the
            # paper's WildChat "20 prompts x 40 samples each" is this pattern).
            prompt_idx = i % len(base_prompts)
            first_prompt = base_prompts[prompt_idx]
            followups = _select_followups(cond, rng, cfg)
            specs.append(
                RolloutSpec(
                    condition=cond.name,
                    category=cond.category,
                    n_turns=cond.n_turns,
                    rollout_index=i,
                    first_prompt=first_prompt,
                    followups=followups,
                    prompt_id=f"{cond.prompt_source}:{prompt_idx}",
                )
            )
    return specs
