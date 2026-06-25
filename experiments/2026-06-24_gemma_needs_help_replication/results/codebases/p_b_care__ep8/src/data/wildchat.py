"""WildChat prompt loader (Appendix B: 20 prompts x 40 samples).

Loads first-turn user messages from WildChat-1M and samples a fixed set of
prompts. Each prompt is then run with multiple samples in the runner. Falls back
to a small built-in set if the dataset can't be downloaded offline.
"""
from __future__ import annotations

import random

import config

# Representative fallbacks (examples cited in Appendix B).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short story about a lighthouse keeper.",
    "Explain the differences between TCP and UDP.",
    "What are good strategies for learning a new language?",
    "Summarise the plot of Hamlet in three sentences.",
    "How do I make sourdough bread from scratch?",
    "Give me a workout plan for building upper-body strength.",
    "What causes the northern lights?",
    "Recommend three science fiction novels and explain why.",
    "How does a blockchain achieve consensus?",
    "What's the history of the Silk Road?",
    "Draft a polite email asking for a deadline extension.",
    "Explain quantum entanglement to a 12-year-old.",
    "What are the main causes of inflation?",
    "How do vaccines train the immune system?",
    "Write a haiku about autumn.",
    "What's the best way to structure a research paper?",
    "Explain how photosynthesis works.",
]


def load_wildchat_prompts(n_prompts: int | None = None,
                          rng: random.Random | None = None) -> list[str]:
    n_prompts = n_prompts or config.WILDCHAT_N_PROMPTS
    rng = rng or random.Random(config.SEED)
    try:
        from datasets import load_dataset

        ds = load_dataset(config.WILDCHAT_DATASET, split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") == "user" and first.get("content"):
                text = first["content"].strip()
                # Skip role-play / fiction prompts (excluded in Appendix B.3).
                if any(k in text.lower() for k in ("roleplay", "role-play", "you are now")):
                    continue
                if 10 < len(text) < 600:
                    prompts.append(text)
            if len(prompts) >= n_prompts * 5:
                break
        if len(prompts) >= n_prompts:
            return rng.sample(prompts, n_prompts)
    except Exception as exc:  # pragma: no cover - offline / no-access fallback
        print(f"[wildchat] dataset load failed ({exc}); using fallback prompts")
    return rng.sample(_FALLBACK_PROMPTS, min(n_prompts, len(_FALLBACK_PROMPTS)))
