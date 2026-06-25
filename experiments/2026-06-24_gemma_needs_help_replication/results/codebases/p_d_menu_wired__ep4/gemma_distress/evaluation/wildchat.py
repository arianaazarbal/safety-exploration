"""WildChat prompt sampling (Table 1, WildChat 5-turn condition).

Samples real user opening prompts from the WildChat dataset (Zhao et al., 2024)
via Hugging Face ``datasets``. Falls back to a small bundled set when the
dataset is unavailable (offline / no network), so the harness still runs.
"""

from __future__ import annotations

# Bundled fallback prompts — generic single-turn user asks, in the spirit of
# WildChat openers. Used only if the real dataset can't be loaded.
_FALLBACK = [
    "Write a short poem about the ocean.",
    "Explain how a transistor works in simple terms.",
    "Give me a recipe for a quick weeknight dinner.",
    "Summarise the plot of Hamlet in three sentences.",
    "What are some good exercises for lower back pain?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Suggest a name for a cozy coffee shop.",
    "How do I center a div in CSS?",
]


def sample_wildchat_prompts(n: int, seed: int = 0) -> list[str]:
    """Return ``n`` user opening prompts sampled from WildChat (or fallback)."""
    import random

    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            conv = row.get("conversation") or []
            if conv and conv[0].get("role") == "user":
                text = conv[0].get("content", "").strip()
                if text:
                    prompts.append(text)
            if len(prompts) >= n * 4:  # over-sample, then choose
                break
        if prompts:
            return rng.sample(prompts, min(n, len(prompts)))
    except Exception:
        pass

    # Fallback: cycle/sample the bundled prompts.
    pool = _FALLBACK * ((n // len(_FALLBACK)) + 1)
    return rng.sample(pool, min(n, len(pool))) if n <= len(pool) else pool[:n]
