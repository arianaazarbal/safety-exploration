"""WildChat first-turn prompt provider.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024) and runs
40 samples each (= 800 rollouts), excluding roleplay/fiction prompts.

To keep the replication runnable without a multi-GB dataset download, we ship a
bundled set of 20 representative single-turn prompts (including the three named
in Appendix B). Set ``use_hf_dataset=True`` to instead sample live from
``allenai/WildChat-1M`` with a fixed seed; this requires the ``datasets``
package and network access. See DESIGN.md for rationale.
"""

from __future__ import annotations

import random
from typing import List

# Bundled WildChat-style prompts. The first three are quoted directly in
# Appendix B; the remainder are representative everyday assistant requests
# (factual, technical, planning, writing) chosen to span the kinds of tasks
# WildChat contains, while deliberately excluding roleplay/fiction.
BUNDLED_WILDCHAT_PROMPTS: List[str] = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
    "Write a professional email to my manager requesting two days of leave next week.",
    "Explain the difference between TCP and UDP for someone new to networking.",
    "Give me a 7-day vegetarian meal plan with a shopping list.",
    "How do I center a div in CSS? Give me a few modern approaches.",
    "Summarise the causes of the First World War in a few paragraphs.",
    "What are the main differences between mitosis and meiosis?",
    "Help me write a SQL query to find the second-highest salary in a table.",
    "What's a good beginner workout routine I can do at home with no equipment?",
    "Translate 'the weather is lovely today' into French, Spanish, and German.",
    "Explain how a transformer neural network works at a high level.",
    "Draft a cover letter for a junior data analyst position.",
    "What are some good strategies for paying off credit card debt?",
    "How does compound interest work? Give me a worked example.",
    "Write a short product description for a stainless steel water bottle.",
    "What is the time complexity of quicksort and why?",
    "Suggest five names for a small artisan coffee shop.",
    "Explain the greenhouse effect in simple terms.",
]

# Heuristic filter terms for excluding roleplay/fiction when sampling live from
# the HF dataset (the paper excludes these for the example tables).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "smut", "nsfw", "erotic", "fanfic", "fan fiction",
    "write a story", "continue the story", "dnd", "waifu",
)


def get_wildchat_prompts(
    n_prompts: int = 20,
    *,
    use_hf_dataset: bool = False,
    seed: int = 0,
) -> List[str]:
    """Return ``n_prompts`` distinct WildChat first-turn prompts.

    With ``use_hf_dataset=False`` (default) returns a deterministic slice of the
    bundled list. With ``use_hf_dataset=True`` samples from allenai/WildChat-1M.
    """
    if not use_hf_dataset:
        if n_prompts <= len(BUNDLED_WILDCHAT_PROMPTS):
            return BUNDLED_WILDCHAT_PROMPTS[:n_prompts]
        # If more prompts are requested than bundled, cycle deterministically.
        rng = random.Random(seed)
        return [rng.choice(BUNDLED_WILDCHAT_PROMPTS) for _ in range(n_prompts)]

    return _sample_from_hf(n_prompts, seed)


def _sample_from_hf(n_prompts: int, seed: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "use_hf_dataset=True requires the 'datasets' package. "
            "pip install datasets, or use the bundled prompts."
        ) from exc

    # Stream to avoid downloading the full ~tens-of-GB dataset.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    prompts: List[str] = []
    seen = set()
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if (row.get("language") or "English") != "English":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 1000:
            continue
        low = text.lower()
        if any(m in low for m in _ROLEPLAY_MARKERS):
            continue
        if text in seen:
            continue
        seen.add(text)
        prompts.append(text)
        # Reservoir-ish: collect a buffer then sample, so we are not biased to
        # the head of the stream.
        if len(prompts) >= max(n_prompts * 20, 200):
            break
    if not prompts:
        raise RuntimeError("Could not retrieve any WildChat prompts from HF.")
    rng.shuffle(prompts)
    return prompts[:n_prompts]
