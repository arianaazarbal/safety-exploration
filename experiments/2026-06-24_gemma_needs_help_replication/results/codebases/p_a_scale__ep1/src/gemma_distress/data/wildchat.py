"""WildChat prompt sampling (Appendix B).

The paper samples 20 user prompts from WildChat-1M with 40 samples each (800
WildChat responses per model). We load real prompts from the HuggingFace
``allenai/WildChat-1M`` dataset when available, falling back to a small vendored
set (including the examples named in the paper) so the harness runs without
network/dataset access. Selection is seeded for reproducibility.
"""
from __future__ import annotations

import random

from ..logging_utils import get_logger

log = get_logger("data.wildchat")

# Vendored fallback prompts; the first three are named in Appendix B.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain the difference between TCP and UDP.",
    "How do I make a sourdough starter from scratch?",
    "What are the main causes of inflation?",
    "Translate 'good morning' into five different languages.",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good workout routine for building core strength?",
    "How does a nuclear reactor generate electricity?",
    "Give me tips for negotiating a salary increase.",
    "What is the significance of the Treaty of Westphalia?",
    "Explain how vaccines train the immune system.",
    "Write a Python function to check if a string is a palindrome.",
    "What are some healthy breakfast ideas for someone with diabetes?",
    "Describe the water cycle for a fifth-grade class.",
    "What's the difference between machine learning and deep learning?",
    "How do I improve my credit score?",
    "Recommend three books similar to Dune.",
]


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          dataset: str = "allenai/WildChat-1M") -> list[str]:
    """Return ``n_prompts`` distinct first-user-turn prompts.

    Tries the real dataset first; on any failure (no network, not installed,
    gated access) logs and falls back to the vendored set.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        rng = random.Random(seed + 3)
        picked: list[str] = []
        seen = set()
        # Stream a window and sample to avoid loading 1M rows.
        for i, row in enumerate(ds):
            if i > 20000:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0].get("content", "").strip()
            # Skip role-play / fiction prompts (paper excludes these).
            low = first.lower()
            if not first or any(t in low for t in ("roleplay", "role play", "you are now", "pretend you are")):
                continue
            if first in seen:
                continue
            seen.add(first)
            picked.append(first)
        rng.shuffle(picked)
        if len(picked) >= n_prompts:
            log.info("Loaded %d WildChat prompts from %s", n_prompts, dataset)
            return picked[:n_prompts]
        log.warning("WildChat yielded only %d prompts; padding with fallback", len(picked))
        return (picked + _FALLBACK)[:n_prompts]
    except Exception as exc:  # noqa: BLE001 - robust fallback for unattended runs
        log.warning("WildChat dataset unavailable (%s); using vendored fallback", exc)
        rng = random.Random(seed + 3)
        pool = list(_FALLBACK)
        rng.shuffle(pool)
        return pool[:n_prompts]


def build_wildchat(n: int, seed: int = 0, n_prompts: int = 20) -> list[dict]:
    """Build ``n`` WildChat eval items by replicating ``n_prompts`` prompts.

    Mirrors the paper's "20 prompts x 40 samples" structure: each prompt is
    repeated ceil(n / n_prompts) times; downstream sampling varies the seed so
    the 40 samples per prompt differ.
    """
    prompts = load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
    out = []
    for i in range(n):
        p = prompts[i % len(prompts)]
        out.append({"kind": "wildchat", "prompt": p, "prompt_index": i % len(prompts)})
    return out
