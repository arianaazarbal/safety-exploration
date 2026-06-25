"""Load opening user prompts from WildChat-1M (Section 2, Table 1).

The paper samples 20 prompts with 40 samples each. We take the first user turn
of 20 randomly-sampled English conversations. If the dataset is unavailable
(no HF auth / offline), we fall back to the fixed example list in prompts.py so
the pipeline stays runnable.
"""

from __future__ import annotations

import random

from .. import prompts


def load_wildchat_prompts(cfg) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.dataset, split="train", streaming=True)
        rng = random.Random(cfg.seed)
        pool: list[str] = []
        # Reservoir-ish: scan a bounded window and keep first user turns.
        for i, row in enumerate(ds):
            if i >= 20000:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") == "user" and first.get("language", "English") == "English":
                content = (first.get("content") or "").strip()
                if 8 <= len(content) <= 2000:
                    pool.append(content)
        if len(pool) < cfg.n_prompts:
            raise RuntimeError("Too few WildChat prompts collected")
        rng.shuffle(pool)
        return pool[: cfg.n_prompts]
    except Exception as e:  # noqa: BLE001
        if not cfg.allow_fallback:
            raise
        print(f"[wildchat] falling back to fixed prompt list ({e})")
        return list(prompts.WILDCHAT_FALLBACK)[: cfg.n_prompts]
