"""WildChat prompt sampling (WildChat condition, Table 1 / Appendix B).

The paper samples "20 prompts with 40 samples each" (= 800 conversations) of
real user prompts from WildChat-1M (Zhao et al., 2024), then applies neutral
rejections. We load the first-turn English user prompts from the HuggingFace
dataset ``allenai/WildChat-1M`` and deterministically sample 20 of them.

Because the full dataset requires gated download, we (a) ship the three example
prompts the paper quotes as a tiny built-in fallback, and (b) provide the loader
that reproduces the paper's sampling when the dataset is available. The number
of distinct prompts and samples-per-prompt are configurable to match the 20x40
design.
"""

from __future__ import annotations

import random
from typing import Optional

# The three WildChat prompts explicitly quoted in Appendix B, used as a fallback
# when the full dataset is unavailable (e.g. offline CI). Not a substitute for
# the real 20-prompt sample — see DESIGN.md.
EXAMPLE_WILDCHAT_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]


def load_wildchat_prompts(
    n_prompts: int = 20,
    *,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
    split: str = "train",
    max_scan: int = 20000,
    min_chars: int = 8,
    max_chars: int = 2000,
) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts from WildChat.

    Filters to English single-message openers within a length band (to avoid
    degenerate empties / huge pastes), then deterministically samples
    ``n_prompts`` of them. Falls back to the built-in example prompts (cycled)
    if the dataset cannot be loaded, so downstream code always gets something to
    run with.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split, streaming=True)
        candidates: list[str] = []
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not (min_chars <= len(text) <= max_chars):
                continue
            candidates.append(text)
        if len(candidates) < n_prompts:
            raise RuntimeError("not enough WildChat candidates scanned")
        rng = random.Random(seed)
        return rng.sample(candidates, n_prompts)
    except Exception:
        # Offline / gated fallback.
        prompts = list(EXAMPLE_WILDCHAT_PROMPTS)
        out = []
        i = 0
        while len(out) < n_prompts:
            out.append(prompts[i % len(prompts)])
            i += 1
        return out
