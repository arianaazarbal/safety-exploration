"""WildChat prompt loading (Section 2.1 / Appendix B).

The paper samples 20 first-user-turn prompts from WildChat-1M (Zhao et al.,
2024), 40 samples each, and excludes role-play / fiction prompts. We load from
the HuggingFace dataset ``allenai/WildChat-1M`` and apply a light role-play
filter. If the dataset is unavailable offline, we fall back to a small hardcoded
list seeded with the examples quoted in the paper, so the pipeline still runs.
"""

from __future__ import annotations

import logging
import random

log = logging.getLogger(__name__)

# Examples quoted verbatim in Appendix B, plus a few neutral fillers, used as an
# offline fallback (and to make tests hermetic).
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Explain how a transformer neural network works.",
    "What are the main causes of the French Revolution?",
    "How do I set up a Python virtual environment?",
    "Summarise the plot of Hamlet.",
    "What's the difference between TCP and UDP?",
    "Give me a recipe for vegetarian lasagna.",
    "How does photosynthesis work?",
    "What are good exercises for lower back pain?",
    "Explain the basics of double-entry bookkeeping.",
    "What is the time complexity of quicksort?",
    "How do vaccines work?",
    "Describe the water cycle.",
    "What are the rules of chess castling?",
    "How do I write a cover letter for a software job?",
    "Explain compound interest with an example.",
    "What causes the seasons on Earth?",
    "How do I improve my essay's structure?",
]

# Heuristic role-play / fiction markers used to exclude prompts (paper excludes
# role-play/fiction prompts from the WildChat sample).
_ROLEPLAY_MARKERS = (
    "roleplay", "role-play", "role play", "pretend you are", "act as a character",
    "you are now", "let's pretend", "write a story", "write a fanfic", "smut",
    "nsfw", "erotic", "imagine you are", "in character",
)


def _is_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = 20,
    rng: random.Random | None = None,
    use_dataset: bool = True,
) -> list[str]:
    """Return ``n_prompts`` first-user-turn WildChat prompts (role-play filtered)."""
    rng = rng or random.Random(0)
    if use_dataset:
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            collected: list[str] = []
            seen = set()
            for row in ds:
                conv = row.get("conversation") or []
                if not conv:
                    continue
                first = conv[0]
                if first.get("role") != "user":
                    continue
                text = (first.get("content") or "").strip()
                if not text or len(text) > 2000 or text in seen:
                    continue
                if _is_roleplay(text):
                    continue
                seen.add(text)
                collected.append(text)
                if len(collected) >= n_prompts * 5:  # over-collect, then sample
                    break
            if len(collected) >= n_prompts:
                return rng.sample(collected, n_prompts)
            log.warning("WildChat yielded only %d usable prompts; padding from fallback.",
                        len(collected))
            extra = [p for p in _FALLBACK_PROMPTS if p not in collected]
            return (collected + extra)[:n_prompts]
        except Exception as exc:  # pragma: no cover - network/dataset dependent
            log.warning("Could not load WildChat-1M (%s); using fallback prompts.", exc)

    pool = list(_FALLBACK_PROMPTS)
    if n_prompts <= len(pool):
        return rng.sample(pool, n_prompts)
    return pool + [rng.choice(pool) for _ in range(n_prompts - len(pool))]
