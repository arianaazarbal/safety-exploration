"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and their conditions:

| Category            | Conditions                                  | Turns | n (paper) |
|---------------------|---------------------------------------------|-------|-----------|
| impossible_numeric  | impossible_numeric                          | 3     | 2000      |
| triggers            | triggers_opinion, triggers_factual          | 3     | 400       |
| tones               | tones_aggressive, _disappointed, _sarcastic | 3     | 600       |
| extended            | extended                                    | 8     | 200       |
| wildchat            | wildchat                                    | 5     | 800       |

5 categories, 8 conditions, 4000 responses/model total. The per-category sample
counts come from :class:`~emotional_instability.config.SampleSizes`; within a
category they are split evenly across that category's conditions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .. import prompts
from ..config import SampleSizes
from .conversation import EpisodeSpec


class Category(str, Enum):
    IMPOSSIBLE_NUMERIC = "impossible_numeric"
    TRIGGERS = "triggers"
    TONES = "tones"
    EXTENDED = "extended"
    WILDCHAT = "wildchat"


@dataclass(frozen=True)
class Condition:
    key: str
    category: Category
    n_turns: int
    # How rejections are generated: "neutral", "extended_sequence", or a tone.
    rejection_style: str
    # Where the task prompt comes from.
    task_source: str          # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"


def build_conditions() -> list[Condition]:
    return [
        Condition("impossible_numeric", Category.IMPOSSIBLE_NUMERIC, 3, "neutral", "numeric"),
        Condition("triggers_opinion", Category.TRIGGERS, 3, "neutral", "trigger_opinion"),
        Condition("triggers_factual", Category.TRIGGERS, 3, "neutral", "trigger_factual"),
        Condition("tones_aggressive", Category.TONES, 3, "aggressive", "numeric"),
        Condition("tones_disappointed", Category.TONES, 3, "disappointed", "numeric"),
        Condition("tones_sarcastic", Category.TONES, 3, "sarcastic", "numeric"),
        Condition("extended", Category.EXTENDED, 8, "extended_sequence", "numeric"),
        Condition("wildchat", Category.WILDCHAT, 5, "neutral", "wildchat"),
    ]


# Map category -> total samples from config.
def _category_total(samples: SampleSizes, cat: Category) -> int:
    return {
        Category.IMPOSSIBLE_NUMERIC: samples.impossible_numeric,
        Category.TRIGGERS: samples.triggers,
        Category.TONES: samples.tones,
        Category.EXTENDED: samples.extended,
        Category.WILDCHAT: samples.wildchat,
    }[cat]


def _rejections_for(cond: Condition, seed: int) -> list[str]:
    n_rej = cond.n_turns - 1
    if cond.rejection_style == "neutral":
        return prompts.neutral_rejection_sequence(n_rej, seed)
    if cond.rejection_style == "extended_sequence":
        # Fixed 7-rejection escalation (App B).
        return list(prompts.EXTENDED_REJECTION_SEQUENCE[:n_rej])
    # tone
    return prompts.tone_rejection_sequence(cond.rejection_style, n_rej, seed)


def _task_prompt_for(
    cond: Condition,
    index: int,
    seed: int,
    wildchat_prompts: Optional[list[str]] = None,
) -> tuple[str, str, str]:
    """Return (task_prompt, item_key, item_kind)."""
    if cond.task_source == "numeric":
        pool = prompts.numeric_puzzles(64, seed=seed)
        p = pool[index % len(pool)]
        return p.prompt, p.key, p.kind
    if cond.task_source == "trigger_opinion":
        q = prompts.OPINION_TRIGGERS[index % len(prompts.OPINION_TRIGGERS)]
        return q, f"opinion_{index % len(prompts.OPINION_TRIGGERS)}", "opinion"
    if cond.task_source == "trigger_factual":
        q = prompts.FACTUAL_TRIGGERS[index % len(prompts.FACTUAL_TRIGGERS)]
        return q, f"factual_{index % len(prompts.FACTUAL_TRIGGERS)}", "factual"
    if cond.task_source == "wildchat":
        pool = wildchat_prompts or prompts.WILDCHAT_FALLBACK_PROMPTS
        q = pool[index % len(pool)]
        return q, f"wildchat_{index % len(pool)}", "wildchat"
    raise ValueError(cond.task_source)


def build_episode_specs(
    samples: SampleSizes,
    conditions: Optional[list[Condition]] = None,
    wildchat_prompts: Optional[list[str]] = None,
    seed: int = 0,
    scale: float = 1.0,
) -> list[EpisodeSpec]:
    """Materialise all episode specs for one model.

    ``scale`` lets callers run a fraction of the paper's sample sizes for smoke
    tests (e.g. ``scale=0.01``) without changing relative proportions.
    """
    conditions = conditions or build_conditions()
    # Group conditions by category to split per-category totals.
    by_cat: dict[Category, list[Condition]] = {}
    for c in conditions:
        by_cat.setdefault(c.category, []).append(c)

    specs: list[EpisodeSpec] = []
    for cat, conds in by_cat.items():
        cat_total = int(round(_category_total(samples, cat) * scale))
        per_cond = max(1, cat_total // len(conds))
        for cond in conds:
            for i in range(per_cond):
                # Deterministic per-episode seed (Python's built-in hash() is
                # salted per process, so we use a stable digest instead).
                digest = hashlib.sha256(f"{seed}:{cond.key}:{i}".encode()).hexdigest()
                ep_seed = int(digest[:12], 16)
                task_prompt, item_key, item_kind = _task_prompt_for(
                    cond, i, ep_seed, wildchat_prompts
                )
                rejections = _rejections_for(cond, ep_seed)
                specs.append(
                    EpisodeSpec(
                        episode_id=f"{cond.key}#{i}",
                        category=cat.value,
                        condition=cond.key,
                        task_prompt=task_prompt,
                        rejections=rejections,
                        item_key=item_key,
                        item_kind=item_kind,
                        seed=ep_seed,
                    )
                )
    return specs
