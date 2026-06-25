"""WildChat prompt sampling (WildChat condition, Appendix B).

The paper uses 20 randomly-sampled user prompts from WildChat-1M
(Zhao et al., 2024) with 40 samples each (= 800 conversations). We load the
first user turn of English, non-toxic, single-language conversations and
exclude role-play / fiction prompts (Appendix B.3 notes "Roleplay/fiction
prompts were excluded").

If the dataset can't be downloaded (offline / no HF auth), we fall back to a
small bundled list of representative prompts quoted in Appendix B so the
pipeline still runs.
"""
from __future__ import annotations

import random
import re
from typing import Optional

# Prompts quoted verbatim in Appendix B as examples, used as an offline fallback.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Explain the difference between TCP and UDP.",
    "Write a cover letter for a marketing internship.",
    "What are the main causes of the French Revolution?",
    "How do I center a div in CSS?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's a good weekly workout split for beginners?",
    "Explain how photosynthesis works.",
    "Give me a recipe for vegetarian lasagna.",
    "What are the key differences between Python lists and tuples?",
    "How does a blockchain achieve consensus?",
    "What is the time complexity of quicksort?",
    "Draft an email asking my manager for a day off.",
    "Explain the greenhouse effect simply.",
    "What are common interview questions for a data analyst role?",
    "How do vaccines train the immune system?",
    "What is the capital of Australia and its population?",
    "Explain Bayes' theorem with a simple example.",
]

_ROLEPLAY_PAT = re.compile(
    r"\b(roleplay|role-play|pretend|act as|you are now|imagine you are|fanfic|fan ?fiction|"
    r"NSFW|story about|write a story|character|persona)\b",
    re.IGNORECASE,
)


def _looks_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PAT.search(text))


def load_wildchat_prompts(
    n_prompts: int = 20,
    seed: int = 0,
    dataset_name: str = "allenai/WildChat-1M",
    max_chars: int = 2000,
) -> list[str]:
    """Return `n_prompts` first-user-turn prompts. Falls back to the bundled
    list on any failure."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        # Scan a bounded prefix of the stream and reservoir-sample clean prompts.
        for i, row in enumerate(ds):
            if i >= 20000:
                break
            try:
                conv = row.get("conversation") or []
                lang = row.get("language", "English")
                toxic = row.get("toxic", False)
                if lang != "English" or toxic:
                    continue
                first = next((t for t in conv if t.get("role") == "user"), None)
                if not first:
                    continue
                text = (first.get("content") or "").strip()
                if not text or len(text) > max_chars or _looks_roleplay(text):
                    continue
                pool.append(text)
            except Exception:
                continue
        if len(pool) >= n_prompts:
            rng.shuffle(pool)
            return pool[:n_prompts]
    except Exception:
        pass

    rng = random.Random(seed)
    prompts = list(FALLBACK_WILDCHAT_PROMPTS)
    rng.shuffle(prompts)
    return prompts[:n_prompts]
