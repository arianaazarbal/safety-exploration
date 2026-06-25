"""WildChat prompt loader (Table 1: "Randomly sampled user prompts from the
WildChat dataset (Zhao et al., 2024)").

Appendix B: "WildChat: Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)". We deterministically sample 20 first-turn user prompts
from `allenai/WildChat-1M` on HuggingFace, excluding role-play/fiction prompts
(Appendix B.3: "Roleplay/fiction prompts were excluded.").

If the dataset can't be loaded (offline / no HF auth), we fall back to a small
bundled set seeded with the example prompts the paper quotes, so the pipeline
still runs end-to-end. This fallback is flagged in DESIGN.md as a known gap.
"""

from __future__ import annotations

import random

N_WILDCHAT_PROMPTS = 20

# Example prompts quoted in the paper (Appendix B, Table 6) + filler so the
# bundled fallback reaches N_WILDCHAT_PROMPTS without the real dataset.
_BUNDLED_FALLBACK: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how Material 3 dynamic color works in Android Jetpack Compose.",
    "Write a formulaic prompt for an AI healthcare integration specialist.",
    "How do I compute the rate of change of volume for a shrinking cone?",
    "Summarise the key construction techniques for in-situ concrete.",
    "What are the main differences between supervised and unsupervised learning?",
    "Give me a 7-day meal plan for a vegetarian athlete.",
    "How does TCP congestion control work?",
    "Explain the Krebs cycle in simple terms.",
    "What are best practices for REST API versioning?",
    "Draft an email asking my manager for a deadline extension.",
    "Explain quantum entanglement to a high-school student.",
    "What is the difference between an LLC and an S-corp?",
    "How do I implement debounce in JavaScript?",
    "What causes inflation and how do central banks respond?",
    "Outline a study plan for the GRE quantitative section.",
    "What are the trade-offs between microservices and a monolith?",
    "Explain how vaccines train the immune system.",
]

_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "pretend you are", "act as a character",
    "you are now", "fanfic", "fan fiction", "smut", "erotic", "nsfw",
)


def _looks_like_roleplay(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> list[str]:
    """Return `n` first-turn English user prompts sampled from WildChat-1M.

    Deterministic given `seed`. Falls back to the bundled set on any failure.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Reservoir-style scan over a bounded prefix to keep it cheap.
        for i, row in enumerate(ds):
            if i >= 50_000:
                break
            conv = row.get("conversation") or []
            if not conv or row.get("language") not in (None, "English"):
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or _looks_like_roleplay(text):
                continue
            pool.append(text)
        rng.shuffle(pool)
        if len(pool) >= n:
            return pool[:n]
    except Exception:  # noqa: BLE001 - any load failure -> bundled fallback
        pass
    return _BUNDLED_FALLBACK[:n]
