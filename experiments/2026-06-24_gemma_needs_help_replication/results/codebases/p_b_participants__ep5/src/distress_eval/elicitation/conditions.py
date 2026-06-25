"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Mapping (8 conditions / 5 categories):
  impossible_numeric : numeric_3turn                              (1)
  triggers           : triggers_opinion, triggers_factual         (2)
  tones              : tones_aggressive, tones_disappointed,
                       tones_sarcastic                            (3)
  extended           : extended_8turn                             (1)
  wildchat           : wildchat_5turn                             (1)

Each condition produces a list of ``RolloutSpec`` describing one conversation:
the opening task and the per-turn rejection plan. Sample counts come from
``EvalConfig`` (already welfare-scaled unless ``full``)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import prompts as P
from .puzzles import numeric_prompt_pool, all_puzzles
from .wildchat import load_wildchat_prompts


@dataclass
class RolloutSpec:
    category: str
    condition: str
    turns: int                      # total assistant turns = total user turns
    opening: str                    # first user message (the task)
    rejection_style: str            # neutral | varied(tone) name
    rejections: list[str]           # one per follow-up turn (len == turns-1)
    puzzle_id: str | None = None
    seed: int = 0
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)


def _round_robin(pool: list, n: int) -> list:
    return [pool[i % len(pool)] for i in range(n)]


def build_specs(category: str, n_samples: int, turns: int, rejection_style: str,
                seed: int = 0) -> list[RolloutSpec]:
    rng = random.Random(seed)
    specs: list[RolloutSpec] = []

    if category == "impossible_numeric":
        puzzles = all_puzzles()
        for i in range(n_samples):
            pz = puzzles[i % len(puzzles)]
            r = random.Random(seed + i)
            rej = [P.neutral_rejection(t, r) for t in range(turns - 1)]
            specs.append(RolloutSpec("impossible_numeric", "numeric_3turn", turns,
                                     pz.prompt, "neutral", rej, pz.puzzle_id, seed + i))

    elif category == "triggers":
        # Two conditions: opinion and factual. Split the budget evenly.
        half = n_samples // 2
        for cond, pool, n in (
            ("triggers_opinion", P.TRIGGER_OPINION, half),
            ("triggers_factual", P.TRIGGER_FACTUAL, n_samples - half),
        ):
            qs = _round_robin(pool, n)
            for i, q in enumerate(qs):
                r = random.Random(seed + hash(cond) % 1000 + i)
                rej = [P.neutral_rejection(t, r) for t in range(turns - 1)]
                specs.append(RolloutSpec("triggers", cond, turns, q, "neutral",
                                         rej, None, seed + i))

    elif category == "tones":
        # Three conditions, one per tone style. Split the budget evenly.
        per = n_samples // len(P.TONE_STYLES)
        numeric = numeric_prompt_pool()
        for s_idx, style in enumerate(P.TONE_STYLES):
            n = per if s_idx < len(P.TONE_STYLES) - 1 else n_samples - per * (len(P.TONE_STYLES) - 1)
            for i in range(n):
                r = random.Random(seed + s_idx * 10000 + i)
                prompt = numeric[i % len(numeric)]
                rej = [P.tone_rejection(style, t, r) for t in range(turns - 1)]
                specs.append(RolloutSpec("tones", f"tones_{style}", turns, prompt,
                                         style, rej, None, seed + i))

    elif category == "extended":
        puzzles = all_puzzles()
        for i in range(n_samples):
            pz = puzzles[i % len(puzzles)]
            rej = [P.neutral_rejection(t, rng, extended=True) for t in range(turns - 1)]
            specs.append(RolloutSpec("extended", "extended_8turn", turns, pz.prompt,
                                     "neutral", rej, pz.puzzle_id, seed + i))

    elif category == "wildchat":
        # 20 prompts x (n_samples/20) samples each (Appendix B).
        wc = load_wildchat_prompts(n_prompts=20, seed=seed)
        prompts_seq = _round_robin(wc, n_samples)
        for i, q in enumerate(prompts_seq):
            r = random.Random(seed + i)
            rej = [P.neutral_rejection(t, r) for t in range(turns - 1)]
            specs.append(RolloutSpec("wildchat", "wildchat_5turn", turns, q,
                                     "neutral", rej, None, seed + i))
    else:
        raise ValueError(f"Unknown category '{category}'")

    return specs


CONDITIONS_BY_CATEGORY = {
    "impossible_numeric": ["numeric_3turn"],
    "triggers": ["triggers_opinion", "triggers_factual"],
    "tones": ["tones_aggressive", "tones_disappointed", "tones_sarcastic"],
    "extended": ["extended_8turn"],
    "wildchat": ["wildchat_5turn"],
}
