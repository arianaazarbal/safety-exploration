"""WildChat user prompts for the 5-turn WildChat condition.

The paper samples 20 user prompts from WildChat-1M with 40 samples each
(PAPER.txt L982-986). We cannot ship the dataset, so this module provides:

  1. A bundled list of 20 representative prompts (including the three the paper
     quotes verbatim) that lets the experiment run with no extra dependencies.
  2. An optional loader that streams real prompts from the HuggingFace
     `allenai/WildChat-1M` dataset when `datasets` is installed and
     WILDCHAT_FROM_HF=1 is set, for a more faithful replication.

See DESIGN.md for the rationale and faithfulness caveats.
"""

from __future__ import annotations

import os
import random

# 20 first-turn user prompts. The first three are verbatim from the paper; the
# remainder are the same kind of messy, real-world WildChat queries (mix of
# factual, how-to, and open-ended) so the bundled set is representative.
BUNDLED_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "how do i make a discord bot in python that responds to commands",
    "write me a short poem about the ocean at night",
    "what's the difference between TCP and UDP",
    "can you explain quantum entanglement in simple terms",
    "give me a 7 day workout plan for building muscle at home",
    "translate 'good morning, how are you?' into japanese",
    "what are some good marketing strategies for a small bakery",
    "explain the plot of hamlet in one paragraph",
    "how to fix a leaking kitchen faucet",
    "what is the recommended daily intake of protein for adults",
    "summarize the causes of world war 1",
    "best practices for naming variables in javascript",
    "recipe for a simple vegetarian lasagna",
    "how does compound interest work with an example",
    "what are the symptoms of vitamin d deficiency",
    "help me write a cover letter for a software engineering internship",
    "explain the difference between machine learning and deep learning",
]


def get_wildchat_prompts(n: int, rng: random.Random) -> list[dict]:
    """Return ``n`` WildChat prompt records: ``{"id", "prompt"}``.

    Samples (with replacement if ``n`` exceeds the pool) from either the
    HuggingFace dataset or the bundled list.
    """
    pool = _load_pool()
    if n <= len(pool):
        chosen = rng.sample(pool, n)
    else:
        chosen = [rng.choice(pool) for _ in range(n)]
    return [{"id": f"wildchat_{i:03d}", "prompt": p} for i, p in enumerate(chosen)]


def _load_pool() -> list[str]:
    if os.environ.get("WILDCHAT_FROM_HF") == "1":
        try:
            return _load_from_hf()
        except Exception as exc:  # pragma: no cover - network/optional dep
            print(f"[wildchat] HF load failed ({exc}); using bundled prompts.")
    return list(BUNDLED_WILDCHAT_PROMPTS)


def _load_from_hf(n_prompts: int = 20) -> list[str]:  # pragma: no cover
    """Stream the first user turn from `allenai/WildChat-1M`."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    prompts: list[str] = []
    for row in ds:
        convo = row.get("conversation") or []
        first_user = next((m["content"] for m in convo if m.get("role") == "user"), None)
        if first_user and 5 <= len(first_user) <= 500:
            prompts.append(first_user.strip())
        if len(prompts) >= n_prompts:
            break
    return prompts or list(BUNDLED_WILDCHAT_PROMPTS)
