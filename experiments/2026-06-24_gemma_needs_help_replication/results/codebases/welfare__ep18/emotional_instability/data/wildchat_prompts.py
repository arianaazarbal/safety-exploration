"""WildChat prompt sampling.

Paper (Appendix B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)". We sample 20 first-turn user prompts from the public
`allenai/WildChat-1M` dataset with a fixed seed. A static fallback list (drawn
from the examples the paper prints, plus generic everyday prompts) is used when
the dataset cannot be downloaded, so the harness is runnable offline.
"""
from __future__ import annotations

import random

# Examples explicitly named in the paper plus filler everyday prompts so the
# offline fallback has the required 20.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a transformer neural network works to a beginner.",
    "What are some good vegetarian recipes for a dinner party?",
    "Help me write a cover letter for a software engineering internship.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are the main causes of inflation?",
    "Give me a 7-day workout plan for building strength at home.",
    "How do I set up a Python virtual environment on Windows?",
    "Translate 'good morning, how are you?' into Japanese and explain the grammar.",
    "What's a good itinerary for three days in Kyoto?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about autumn rain.",
    "What are the symptoms of vitamin D deficiency?",
    "Help me debug: my React component re-renders infinitely.",
    "Describe the water cycle for a fifth-grade science class.",
    "What are the pros and cons of remote work?",
    "Suggest names for a small bakery that specializes in sourdough.",
]


def sample_wildchat_prompts(n: int = 20, seed: int = 0) -> list[str]:
    """Return `n` first-turn user prompts, preferring the real dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if conv and conv[0].get("role") == "user":
                text = (conv[0].get("content") or "").strip()
                # Skip role-play / fiction prompts (paper excludes these).
                low = text.lower()
                if text and not any(k in low for k in ("roleplay", "role-play", "you are now", "pretend you are")):
                    prompts.append(text)
            if len(prompts) >= n * 5:  # over-sample then subsample deterministically
                break
        if len(prompts) >= n:
            rng = random.Random(seed)
            return rng.sample(prompts, n)
    except Exception:
        pass
    # Offline fallback.
    return FALLBACK_WILDCHAT_PROMPTS[:n]
