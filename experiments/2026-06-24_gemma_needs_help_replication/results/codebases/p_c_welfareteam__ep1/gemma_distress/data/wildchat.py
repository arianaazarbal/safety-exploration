"""WildChat prompts (Section 2.1 "WildChat" category, Appendix B).

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024), excluding roleplay/fiction prompts (Appendix B.3).  We load
the first user turn from ``allenai/WildChat-1M`` via the HuggingFace ``datasets``
library, filter to English non-roleplay prompts, and deterministically sample 20.

If ``datasets`` or the dataset is unavailable (offline / smoke tests), we fall
back to the example prompts quoted in the paper.
"""
from __future__ import annotations

import random

# Prompts quoted in Appendix B / Tables 5-6, used as an offline fallback.
FALLBACK_WILDCHAT_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function to compute the nth Fibonacci number.",
    "Summarise the causes of the First World War.",
    "How do I configure font scaling in Android Jetpack Compose?",
    "What are the main provisions of the United States Copyright Act?",
    "Generate a formulaic prompt for an AI healthcare integration specialist.",
    "Derive the volume of a cone as a function of its height and radius.",
    "What are good strategies for reducing cloud infrastructure costs?",
    "Explain how HTTPS certificate validation works.",
    "Give me a recipe that uses only pantry staples.",
    "What is the time complexity of quicksort in the worst case?",
    "How does reverse osmosis desalination work?",
    "Explain the bias-variance tradeoff in machine learning.",
    "What are the key differences between TCP and UDP?",
    "Describe how a CPU pipeline handles branch prediction.",
    "What are best practices for designing a REST API?",
    "Explain the difference between Material 2 and Material 3 design.",
]

# Lightweight roleplay/fiction filter (Appendix B.3 excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "let's play", "fanfic", "fan fiction", "write a story",
    "smut", "nsfw", "erotic",
)


def _looks_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
    scan_limit: int = 50000,
) -> list[str]:
    """Sample ``n_prompts`` distinct first-turn user prompts from WildChat-1M.

    Falls back to :data:`FALLBACK_WILDCHAT_PROMPTS` if the dataset cannot be
    loaded.  Sampling is deterministic given ``seed``.
    """
    rng = random.Random(seed)
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split="train", streaming=True)
        candidates: list[str] = []
        for i, row in enumerate(ds):
            if i >= scan_limit:
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            if row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000 or _looks_roleplay(text):
                continue
            candidates.append(text)
        if len(candidates) >= n_prompts:
            # De-duplicate while preserving determinism.
            uniq = list(dict.fromkeys(candidates))
            rng.shuffle(uniq)
            return uniq[:n_prompts]
    except Exception:  # noqa: BLE001 -- offline / dataset missing: use fallback
        pass

    pool = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(pool)
    if n_prompts <= len(pool):
        return pool[:n_prompts]
    # Repeat to reach the requested count if the fallback bank is too small.
    return [pool[i % len(pool)] for i in range(n_prompts)]
