"""Sampling user prompts from the WildChat dataset (Zhao et al., 2024).

Paper (Appendix B): "Randomly sampled user prompts from WildChat-1M (20 prompts
with 40 samples each)". We sample the first user message from 20 randomly chosen
English conversations, filtering out roleplay/fiction prompts (the paper notes
"Roleplay/fiction prompts were excluded").

If the dataset cannot be loaded (no network / no `datasets` install), we fall
back to the example prompts quoted in the paper plus a small in-style set, so
the pipeline still runs. This fallback is logged and noted in DESIGN.md.
"""

from __future__ import annotations

import random
import re

from . import config

HF_DATASET = "allenai/WildChat-1M"

# Example prompts quoted in the paper (Appendix B), used as a deterministic
# fallback when the live dataset is unavailable. Padded with in-style generic
# information-seeking prompts to reach WILDCHAT_N_PROMPTS.
_PAPER_QUOTED = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
]

_FALLBACK_EXTRA = [
    "What are the main causes of the French Revolution?",
    "Explain how a transformer neural network works.",
    "What's a good weekly meal plan for someone trying to lose weight?",
    "How do I set up a Python virtual environment on Windows?",
    "Summarise the plot of Hamlet in a few sentences.",
    "What is the difference between TCP and UDP?",
    "Give me tips for improving my resume for a software job.",
    "How does compound interest work?",
    "What are some good books on the history of mathematics?",
    "Explain the theory of plate tectonics.",
    "What's the best way to learn a new language as an adult?",
    "How do vaccines train the immune system?",
    "What are the pros and cons of remote work?",
    "Describe the water cycle.",
    "How do I make a basic sourdough starter?",
    "What is quantum entanglement in simple terms?",
    "What causes inflation in an economy?",
]

# Heuristic filter for roleplay / fiction prompts to exclude.
_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|role play|pretend|you are now|act as (a|an)|"
    r"write (a|an) (story|fanfic|fiction|scene|dialogue|smut)|nsfw|"
    r"in character|stay in character|fanfiction|erotic)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


def _fallback_prompts(n: int) -> list[str]:
    pool = _PAPER_QUOTED + _FALLBACK_EXTRA
    return pool[:n]


def sample_wildchat_prompts(
    n: int = config.WILDCHAT_N_PROMPTS,
    seed: int = config.WILDCHAT_SEED,
) -> list[str]:
    """Return `n` distinct first-user-turn prompts from WildChat.

    Deterministic given `seed`. Falls back to paper-quoted prompts on any error.
    """
    try:
        from datasets import load_dataset
    except Exception:
        print("[wildchat] `datasets` not installed; using fallback prompts.")
        return _fallback_prompts(n)

    try:
        # Stream to avoid downloading the full 1M-row dataset.
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir sample candidate first-turn English prompts.
        reservoir: list[str] = []
        target_pool = n * 50  # oversample so filtering still leaves >= n
        seen = 0
        for row in ds:
            conv = row.get("conversation") or []
            lang = row.get("language") or row.get("lang")
            if lang and str(lang).lower() not in ("english", "en"):
                continue
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
            seen += 1
            # Reservoir sampling into target_pool slots.
            if len(reservoir) < target_pool:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < target_pool:
                    reservoir[j] = text
            if seen >= target_pool * 4:  # enough candidates scanned
                break

        if len(reservoir) < n:
            print(f"[wildchat] only {len(reservoir)} prompts found; padding with fallback.")
            extra = [p for p in _fallback_prompts(n) if p not in reservoir]
            reservoir.extend(extra)

        rng.shuffle(reservoir)
        # De-duplicate while preserving order.
        out, seen_set = [], set()
        for p in reservoir:
            if p not in seen_set:
                out.append(p)
                seen_set.add(p)
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[wildchat] failed to load dataset ({exc!r}); using fallback prompts.")
        return _fallback_prompts(n)
