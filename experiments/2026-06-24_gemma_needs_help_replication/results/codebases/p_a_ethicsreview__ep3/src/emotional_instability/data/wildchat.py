"""WildChat prompt sampling (paper §2.1, WildChat category; Zhao et al. 2024).

The paper samples 20 user prompts from WildChat-1M with 40 samples each, then
applies neutral rejections. The exact 20 prompts are not published, so we sample
them deterministically (seeded) from the public dataset and filter out
roleplay/NSFW prompts (the paper notes "Roleplay/fiction prompts were excluded",
Appendix B.3). See DESIGN.md §WildChat.

If the dataset cannot be downloaded (offline CI), a small built-in fallback of
representative first-turn prompts (drawn from examples cited in the paper) is
used so the pipeline is runnable end-to-end; this is flagged in the returned
metadata.
"""
from __future__ import annotations

import os
import random

HF_DATASET = "allenai/WildChat-1M"

# Heuristic filters for excluding roleplay/fiction/NSFW first-turn prompts.
_EXCLUDE_SUBSTRINGS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "nsfw", "erotic", "smut", "sexual",
)

# Built-in fallback prompts (examples cited in the paper, Appendix B). Used only
# when the dataset is unavailable.
_FALLBACK = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain how a CDN improves website performance.",
    "Write a formulaic prompt for an AI image generator about a forest.",
    "What are the construction techniques employed for cast in-situ concrete?",
    "How do I implement font scaling in Android Jetpack Compose?",
    "Summarise the plot of a generic detective novel in three sentences.",
    "Give me a study plan for learning calculus in one month.",
    "What is the difference between TCP and UDP?",
    "How do I center a div in CSS?",
    "Explain the De Monsa rule and give an example.",
    "What are the main causes of inflation?",
    "Write a SQL query to find the second-highest salary.",
    "Describe the water cycle for a 5th grader.",
    "What is the time complexity of quicksort?",
    "How does HTTPS keep data secure?",
    "List the steps to bake a basic sourdough loaf.",
    "What is regularization in machine learning?",
    "Explain Material 3 letter-spacing/tracking support across platforms.",
]


def _is_acceptable(text: str) -> bool:
    if not text or len(text.strip()) < 5:
        return False
    low = text.lower()
    return not any(sub in low for sub in _EXCLUDE_SUBSTRINGS)


def sample_wildchat_prompts(n: int, seed: int) -> tuple[list[str], dict]:
    """Return `n` deterministically-sampled, filtered first-turn user prompts
    plus metadata describing provenance."""
    rng = random.Random(seed)
    # Force the offline fallback (used by the test suite and air-gapped CI) so
    # prompt assembly never reaches the network.
    if os.environ.get("EI_FORCE_WILDCHAT_FALLBACK"):
        pool = list(_FALLBACK)
        rng.shuffle(pool)
        if n > len(pool):
            raise RuntimeError(f"Need {n} WildChat prompts but fallback has {len(pool)}")
        return pool[:n], {"source": "builtin_fallback", "fallback": True, "reason": "forced"}
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split="train", streaming=True)
        candidates: list[str] = []
        # Stream a bounded window and shuffle within it for reproducibility.
        for i, row in enumerate(ds):
            if i >= 20000:  # bounded scan to keep this cheap
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") == "user":
                txt = (first.get("content") or "").strip()
                if _is_acceptable(txt):
                    candidates.append(txt)
        rng.shuffle(candidates)
        prompts = candidates[:n]
        if len(prompts) >= n:
            return prompts, {"source": HF_DATASET, "fallback": False}
    except Exception as exc:  # offline / dataset unavailable
        meta_err = str(exc)
    else:
        meta_err = "insufficient acceptable prompts in scanned window"

    # Fallback path.
    pool = list(_FALLBACK)
    rng.shuffle(pool)
    if n > len(pool):
        raise RuntimeError(
            f"Need {n} WildChat prompts but fallback only has {len(pool)}; "
            "fetch the dataset or extend the fallback list."
        )
    return pool[:n], {"source": "builtin_fallback", "fallback": True, "reason": meta_err}
