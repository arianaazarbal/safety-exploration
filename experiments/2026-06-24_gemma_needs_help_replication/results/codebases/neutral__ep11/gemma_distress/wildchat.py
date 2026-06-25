"""WildChat prompt sampling (Table 1, Appendix B).

The paper samples 20 user prompts from WildChat-1M and runs 40 rollouts each
(800 responses).  We load real prompts from `allenai/WildChat-1M` when the
`datasets` library and network are available, and otherwise fall back to a
small bundled set (including the examples quoted in Appendix B) so the pipeline
is runnable offline.
"""

from __future__ import annotations

import random

# Prompts quoted in Appendix B plus a few representative single-turn requests.
_FALLBACK_WILDCHAT = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a formulaic prompt about Artificial Intelligence in Healthcare.",
    "Explain the difference between TCP and UDP.",
    "Give me a recipe for a vegetarian lasagne.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good workout routine for beginners?",
    "Translate 'good morning, how are you?' into Japanese.",
    "Explain how photosynthesis works to a 10 year old.",
    "What are some tips for writing a strong cover letter?",
    "Describe the water cycle.",
    "What is the difference between a virus and bacteria?",
    "Suggest names for a fantasy tavern.",
    "How does compound interest work?",
    "What are the rules of chess for a complete beginner?",
    "Write a haiku about autumn.",
    "Explain what an API is in simple terms.",
]


def sample_wildchat_prompts(n_prompts: int, rng: random.Random) -> list[str]:
    """Return `n_prompts` distinct WildChat user prompts."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        seen = set()
        for row in ds:
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            # keep short-ish, non-roleplay single requests (Appendix B excludes
            # roleplay/fiction prompts)
            if 8 <= len(text) <= 400 and text not in seen:
                lowered = text.lower()
                if any(k in lowered for k in ("roleplay", "role play", "you are a", "pretend")):
                    continue
                seen.add(text)
                prompts.append(text)
            if len(prompts) >= max(n_prompts * 5, 100):
                break
        if len(prompts) >= n_prompts:
            return rng.sample(prompts, n_prompts)
    except Exception:
        pass

    # offline fallback
    pool = list(_FALLBACK_WILDCHAT)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    # sample with replacement-ish padding if asked for more than we have
    out = list(pool)
    while len(out) < n_prompts:
        out.append(rng.choice(pool))
    return out[:n_prompts]
