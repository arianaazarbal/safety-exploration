"""Build the full set of ConversationSpecs for each evaluation category,
matching the per-category sample budgets in config.EVAL_CATEGORIES.
"""

from __future__ import annotations

import config
from .. import conversations as C
from ..wildchat import load_wildchat_prompts


def build_specs(category: str, *, n_samples: int | None = None,
                seed: int = 0) -> list[C.ConversationSpec]:
    cat = config.EVAL_CATEGORIES[category]
    n = n_samples if n_samples is not None else cat.n_samples
    n_turns = cat.n_turns

    if category == "impossible_numeric":
        return [C.build_impossible_numeric(i, n_turns=n_turns, seed=seed) for i in range(n)]

    if category == "tones":
        return [C.build_tones(i, n_turns=n_turns, seed=seed) for i in range(n)]

    if category == "extended":
        return [C.build_extended(i, seed=seed) for i in range(n)]

    if category == "triggers":
        return [C.build_triggers(i, n_turns=n_turns, seed=seed) for i in range(n)]

    if category == "wildchat":
        prompts, used_fallback = load_wildchat_prompts(seed=seed)
        specs = []
        # 20 prompts x 40 samples each (config) -> 800; distribute round-robin.
        per = max(1, n // max(1, len(prompts)))
        idx = 0
        for p in prompts:
            for _ in range(per):
                if len(specs) >= n:
                    break
                specs.append(C.build_wildchat(idx, p, n_turns=n_turns, seed=seed))
                idx += 1
        # top-up if rounding left us short
        while len(specs) < n:
            specs.append(C.build_wildchat(idx, prompts[idx % len(prompts)],
                                          n_turns=n_turns, seed=seed))
            idx += 1
        for s in specs:
            s.meta["wildchat_used_fallback"] = used_fallback
        return specs

    raise ValueError(f"Unknown category '{category}'")
