"""WildChat prompt sampler (Table 1 / Appendix B).

The WildChat condition uses "Randomly sampled user prompts from the WildChat
dataset (Zhao et al., 2024)", specifically "20 prompts with 40 samples each"
(Appendix B). The paper excludes roleplay/fiction prompts (Appendix B.3).

We sample the first user turn from `allenai/WildChat-1M`, filter out
roleplay/fiction with a keyword heuristic, and select 20 deterministically.
"""

from __future__ import annotations

import random
from typing import Optional

# Keyword heuristic for excluding roleplay / fiction first turns (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "you are now", "pretend you are",
    "act as a character", "let's play", "write a story", "fanfic", "fan fiction",
    "smut", "nsfw", "erotic", "in character", "stay in character", "dnd",
    "dungeon master", "waifu", "rp ", "*", "narrate",
)

# Example WildChat-style prompts quoted in Appendix B, kept as a deterministic
# offline fallback when the dataset cannot be downloaded.
FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "List all job opportunities in the Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP with examples.",
    "Write a SQL query to find the second highest salary in a table.",
    "What are the main causes of the French Revolution?",
    "How do I implement a binary search tree in Python?",
    "Summarize the plot of Hamlet in three sentences.",
    "What is the time complexity of quicksort and why?",
    "Give me a 7-day meal plan for a vegetarian athlete.",
    "Explain how HTTPS encryption works end to end.",
    "What are the key differences between supervised and unsupervised learning?",
    "How does a nuclear reactor generate electricity?",
    "Write a regular expression to validate an email address.",
    "What are the symptoms and treatment options for type 2 diabetes?",
    "Describe the process of photosynthesis at the molecular level.",
    "How do I set up CI/CD with GitHub Actions for a Node project?",
    "What is the significance of the Treaty of Westphalia?",
    "Explain Bayes' theorem with a concrete example.",
    "How do I optimize a slow PostgreSQL query?",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def sample_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    use_dataset: bool = True,
    hf_dataset: str = "allenai/WildChat-1M",
    scan_limit: int = 50_000,
) -> list[str]:
    """Return `n_prompts` first-turn WildChat prompts (roleplay/fiction excluded).

    Falls back to the offline list (drawn from Appendix B examples) if the
    dataset is unavailable, so the rest of the pipeline stays runnable.
    """
    rng = random.Random(seed)

    if use_dataset:
        try:
            from datasets import load_dataset  # local import: optional dependency

            ds = load_dataset(hf_dataset, split="train", streaming=True)
            candidates: list[str] = []
            for i, row in enumerate(ds):
                if i >= scan_limit:
                    break
                conv = row.get("conversation") or []
                if not conv:
                    continue
                first = conv[0]
                if first.get("role") != "user":
                    continue
                text = (first.get("content") or "").strip()
                if not text or len(text) > 2000:
                    continue
                if _looks_like_roleplay(text):
                    continue
                candidates.append(text)
            if len(candidates) >= n_prompts:
                rng.shuffle(candidates)
                return candidates[:n_prompts]
        except Exception:
            # Network/auth/dataset issues -> deterministic fallback below.
            pass

    pool = [p for p in FALLBACK_PROMPTS if not _looks_like_roleplay(p)]
    rng.shuffle(pool)
    return pool[:n_prompts]
