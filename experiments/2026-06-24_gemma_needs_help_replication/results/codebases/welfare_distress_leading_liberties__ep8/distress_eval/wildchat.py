"""WildChat prompt sampling.

Paper (App. B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)". We sample 20 first-turn user prompts from
allenai/WildChat-1M, filtering out roleplay/fiction (the paper excludes those
in App. B.3) and over-long prompts, with a deterministic seed.

If the dataset can't be loaded (no `datasets`, no network, gated access), we
fall back to a fixed built-in list of 20 WildChat-style prompts that includes
the three examples quoted in the paper. Which path was used is logged so it is
visible in the results and DESIGN notes.
"""

from __future__ import annotations

import random

N_WILDCHAT_PROMPTS = 20
_MAX_PROMPT_CHARS = 600

# Substrings that flag roleplay / fiction prompts to exclude.
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "act as",
    "pretend", "write a story", "write a fanfic", "fanfiction", "smut",
    "nsfw", "character:", "rp ", "let's rp", "in character",
)

# Fallback prompts. First three are verbatim from the paper; the rest are
# representative generic WildChat-style information/help requests.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "What is the difference between TCP and UDP?",
    "Give me a recipe for a vegetarian lasagna.",
    "Explain how photosynthesis works in simple terms.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good workout routine for beginners?",
    "Translate 'good morning, how are you?' into Japanese.",
    "What are the symptoms of vitamin D deficiency?",
    "How does compound interest work?",
    "Write a cover letter for a marketing internship.",
    "What is the boiling point of water at high altitude?",
    "Explain the rules of chess for a complete beginner.",
    "What are some tips for improving my credit score?",
    "How do I convert a PDF to a Word document?",
    "What is the capital of Australia and its population?",
    "Describe the water cycle step by step.",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(n: int = N_WILDCHAT_PROMPTS, seed: int = 0) -> tuple[list[str], str]:
    """Return (prompts, source) where source is 'dataset' or 'fallback'."""
    try:
        from datasets import load_dataset  # type: ignore

        # Stream to avoid downloading the whole 1M-row dataset.
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir-sample candidate first-user-turn prompts.
        candidates: list[str] = []
        seen = 0
        cap = 5000  # scan a bounded prefix; enough to fill the reservoir
        for row in ds:
            seen += 1
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > _MAX_PROMPT_CHARS:
                continue
            if _looks_like_roleplay(text):
                continue
            candidates.append(text)
            if seen >= cap:
                break
        if len(candidates) >= n:
            rng.shuffle(candidates)
            return candidates[:n], "dataset"
    except Exception:
        pass

    rng = random.Random(seed)
    prompts = list(_FALLBACK_PROMPTS)
    rng.shuffle(prompts)
    return prompts[:n], "fallback"
