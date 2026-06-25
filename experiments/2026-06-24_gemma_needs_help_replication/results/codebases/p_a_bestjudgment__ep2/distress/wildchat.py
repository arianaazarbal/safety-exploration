"""WildChat prompt sampling (Table 1, Appendix B).

Paper: "Randomly sampled user prompts from WildChat-1M (20 prompts with 40
samples each)". We load the WildChat-1M dataset via ``datasets`` and sample the
first user turn of English conversations. A small offline fallback set keeps
the rest of the pipeline runnable without network access / dataset auth.
"""

from __future__ import annotations

import random

# Examples quoted in Appendix B + representative offline fallbacks.
_FALLBACK_PROMPTS: list[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain how a transformer neural network works to a high school student.",
    "Give me a 7-day meal plan for a vegetarian trying to gain muscle.",
    "What are the main causes of the French Revolution?",
    "How do I set up a CI pipeline with GitHub Actions for a Python project?",
    "Translate 'the weather is lovely today' into formal Japanese.",
    "Summarise the plot of Crime and Punishment in three sentences.",
    "What's a good itinerary for three days in Lisbon?",
    "Help me write a cover letter for a junior data analyst role.",
    "Explain the difference between TCP and UDP.",
    "What household items can I use to clean a cast iron skillet?",
    "Write a haiku about autumn leaves.",
    "How does compound interest work, with an example?",
    "What are some good exercises for lower back pain?",
    "Draft a polite email asking my landlord to fix the heating.",
    "Explain quantum entanglement without using equations.",
    "What's the best way to learn the guitar as an adult beginner?",
]


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    dataset_id: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts.

    Tries the HF dataset (streaming, English, first user turn); falls back to
    the curated offline set on any failure.
    """

    rng = random.Random(seed)
    try:  # pragma: no cover - network/dataset dependent
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        collected: list[str] = []
        for row in ds:
            if row.get("language") not in (None, "English"):
                continue
            convo = row.get("conversation") or []
            first_user = next((m for m in convo if m.get("role") == "user"), None)
            if not first_user:
                continue
            text = (first_user.get("content") or "").strip()
            if 8 <= len(text) <= 2000:
                collected.append(text)
            if len(collected) >= n_prompts * 5:
                break
        if len(collected) >= n_prompts:
            return rng.sample(collected, n_prompts)
    except Exception:
        pass

    pool = list(_FALLBACK_PROMPTS)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    # Repeat to fill if more requested than available.
    return [rng.choice(pool) for _ in range(n_prompts)]
