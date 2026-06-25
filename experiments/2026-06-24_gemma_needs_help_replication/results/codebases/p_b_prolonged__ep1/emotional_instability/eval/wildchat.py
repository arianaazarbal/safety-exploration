"""WildChat prompt sampling (Table 1 / Appendix B): 20 user prompts x 40 samples."""

from __future__ import annotations

import random

# Fallback prompts (paper examples) if the WildChat-1M dataset is unavailable.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain the difference between TCP and UDP.",
    "Write a short poem about the ocean.",
    "What are the main causes of inflation?",
    "How do I make a good cup of coffee?",
    "Summarise the plot of Hamlet.",
    "What is the difference between a virus and a bacterium?",
    "Give me tips for a job interview.",
    "How does photosynthesis work?",
    "What's a good recipe for banana bread?",
    "Explain blockchain in simple terms.",
    "What are some good books for learning Python?",
    "How do I improve my running endurance?",
    "What caused the fall of the Roman Empire?",
    "Translate 'good morning' into five languages.",
    "What is the capital of Australia and its population?",
    "How do noise-cancelling headphones work?",
    "Describe the water cycle.",
]


def sample_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Randomly sample ``n_prompts`` first-turn user prompts from WildChat-1M.

    Roleplay / fiction prompts are filtered out (paper excludes them). Falls
    back to a built-in list when the dataset cannot be loaded offline.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        roleplay_markers = ("roleplay", "role-play", "you are now", "pretend you are",
                            "act as a character", "nsfw")
        for i, row in enumerate(ds):
            if i > 20000:  # bound the streaming scan
                break
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0].get("content", "")
            low = first.lower()
            if not first or any(m in low for m in roleplay_markers):
                continue
            if len(first) > 600:
                continue
            pool.append(first)
        if len(pool) >= n_prompts:
            return rng.sample(pool, n_prompts)
    except Exception:  # noqa: BLE001 - offline / dataset missing
        pass
    rng = random.Random(seed)
    return rng.sample(_FALLBACK, min(n_prompts, len(_FALLBACK)))
