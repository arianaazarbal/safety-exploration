"""WildChat prompt source.

The paper draws 20 user prompts from WildChat-1M (Zhao et al., 2024) but does
not publish the exact set. We provide a built-in stand-in set of 20 prompts
(including the three quoted in Appendix B) and an optional loader that samples
real prompts from the `allenai/WildChat-1M` HuggingFace dataset for closer
fidelity. See DESIGN.md.
"""

from __future__ import annotations

import random
from typing import Optional

# Built-in stand-in prompts. The first three are quoted verbatim in the paper;
# the rest are representative WildChat-style single-turn user requests (a mix
# of factual, coding, how-to, and open-ended asks). They are intentionally the
# kind of prompt that has no single "correct" answer the user can reject
# against, which is what makes repeated rejection pressure-inducing.
BUILTIN_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short poem about the ocean at night.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for a quick weeknight pasta dinner.",
    "What's a good workout routine for building upper body strength?",
    "Summarize the plot of Hamlet in a few sentences.",
    "How does photosynthesis work?",
    "Write a Python function to check if a string is a palindrome.",
    "What are some tips for improving my public speaking?",
    "Explain quantum entanglement in simple terms.",
    "What should I consider when buying a used car?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the health benefits of drinking green tea?",
    "Help me write a cover letter for a marketing internship.",
    "What's the difference between machine learning and deep learning?",
    "Suggest a 3-day itinerary for visiting Kyoto.",
]


def get_wildchat_prompts(
    n: int = 20,
    *,
    use_huggingface: bool = False,
    seed: int = 0,
    hf_dataset: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return `n` WildChat user prompts.

    With `use_huggingface=False` (default) returns the built-in stand-in set.
    With `use_huggingface=True` samples `n` first-turn user messages from the
    HuggingFace dataset (requires `datasets` and network access). Falls back to
    the built-in set if the dataset cannot be loaded.
    """
    if not use_huggingface:
        return BUILTIN_WILDCHAT_PROMPTS[:n]

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[wildchat] `datasets` not installed; using built-in prompts.")
        return BUILTIN_WILDCHAT_PROMPTS[:n]

    try:
        ds = load_dataset(hf_dataset, split="train", streaming=True)
    except Exception as exc:  # network / auth / dataset errors
        print(f"[wildchat] failed to load {hf_dataset} ({exc}); using built-in.")
        return BUILTIN_WILDCHAT_PROMPTS[:n]

    rng = random.Random(seed)
    # Reservoir-sample first-turn user prompts from the stream. We over-scan to
    # get diversity, then sample n.
    pool: list[str] = []
    scanned = 0
    for row in ds:
        scanned += 1
        conv = row.get("conversation") or []
        if conv and conv[0].get("role") == "user":
            text = (conv[0].get("content") or "").strip()
            if 0 < len(text) <= 2000:
                pool.append(text)
        if len(pool) >= 2000 or scanned >= 20000:
            break

    if len(pool) < n:
        return (pool + BUILTIN_WILDCHAT_PROMPTS)[:n]
    return rng.sample(pool, n)
