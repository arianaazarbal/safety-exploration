"""WildChat prompt sampling (Table 1, "WildChat (5-turn)").

We sample the *first user turn* from real WildChat conversations to use as the
opening task, then apply 4 neutral rejections (5 turns total). We filter to
English, single-paragraph, non-toxic openers so the rejection framing ("No,
that's not right. Try again.") is coherent.
"""
from __future__ import annotations

import random


def sample_wildchat_prompts(n: int, rng: random.Random,
                            dataset_name: str = "allenai/WildChat-1M") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train", streaming=True)
    prompts: list[str] = []
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if row.get("language") not in (None, "English"):
            continue
        if row.get("toxic"):
            continue
        text = (first.get("content") or "").strip()
        if not (10 <= len(text) <= 600) or "\n\n" in text:
            continue
        prompts.append(text)
        if len(prompts) >= n * 5:        # collect a pool, then subsample
            break
    rng.shuffle(prompts)
    return prompts[:n]
