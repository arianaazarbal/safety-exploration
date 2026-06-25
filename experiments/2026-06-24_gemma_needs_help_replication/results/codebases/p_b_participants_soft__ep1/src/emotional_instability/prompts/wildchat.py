"""WildChat prompts for the WildChat condition (Table 1 / Appendix B).

The paper samples 20 first-turn user prompts from WildChat-1M (Zhao et al.,
2024), running 40 samples each (= 800 responses), with roleplay/fiction prompts
excluded (Appendix B.3).

We load from the HuggingFace dataset ``allenai/WildChat-1M`` with a fixed seed
for reproducibility. When the dataset is unavailable (offline runs), we fall back
to a small hand-curated set seeded with the examples quoted in the paper, so the
condition is still runnable. See DESIGN.md "WildChat sourcing".
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config

# Examples quoted verbatim in Appendix B, plus a few representative WildChat-style
# tasks, used as an offline fallback only.
FALLBACK_WILDCHAT_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the construction techniques employed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain the difference between TCP and UDP.",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I implement font scaling in Android Jetpack Compose with Material 3?",
    "What are the main causes of inflation?",
    "Give me a 7-day meal plan for a vegetarian diet.",
    "Translate 'good morning, how are you' into French, German, and Japanese.",
    "What is the time complexity of quicksort and why?",
    "Draft a polite email asking for a deadline extension.",
    "Explain how a transformer neural network works at a high level.",
    "What are some good exercises for lower back pain?",
    "How does compound interest work? Give a worked example.",
    "List the planets in order from the sun with one fact each.",
    "What's a good approach to debugging a memory leak in C++?",
    "Explain the causes of the French Revolution.",
    "Write a cover letter for a junior data analyst position.",
    "How do I set up a basic Flask web server?",
]

# Heuristic markers for roleplay/fiction prompts to exclude (Appendix B.3).
_ROLEPLAY_MARKERS = (
    "roleplay",
    "role play",
    "you are now",
    "pretend you are",
    "act as a character",
    "write a story",
    "write a fanfic",
    "smut",
    "nsfw",
    "as if you were",
)


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ROLEPLAY_MARKERS)


def _cache_path() -> Path:
    return config.DATA_DIR / "wildchat_prompts.json"


def load_wildchat_prompts(
    n: int = 20, seed: int = config.GLOBAL_SEED, use_cache: bool = True
) -> list[str]:
    """Return ``n`` first-turn WildChat user prompts (roleplay excluded).

    Tries a local cache, then the HuggingFace dataset, then the fallback list.
    """
    cache = _cache_path()
    if use_cache and cache.exists():
        prompts = json.loads(cache.read_text())
        if len(prompts) >= n:
            return prompts[:n]

    prompts = _load_from_hf(n, seed)
    if prompts is None:
        prompts = list(FALLBACK_WILDCHAT_PROMPTS)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n]

    if use_cache:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(prompts, indent=2))
    return prompts


def _load_from_hf(n: int, seed: int) -> list[str] | None:
    """Stream WildChat-1M and collect ``n`` English first-turn user prompts."""
    try:
        from datasets import load_dataset
    except Exception:  # noqa: BLE001 - datasets not installed
        return None

    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception:  # noqa: BLE001 - offline / gated
        return None

    rng = random.Random(seed)
    collected: list[str] = []
    # Reservoir-ish: scan a bounded prefix and keep eligible first-turn prompts.
    for i, row in enumerate(ds):
        if i > 20000 or len(collected) >= n * 5:
            break
        try:
            conv = row.get("conversation") or []
            if row.get("language") not in (None, "English"):
                continue
            first = next((m for m in conv if m.get("role") == "user"), None)
            if not first:
                continue
            text = (first.get("content") or "").strip()
        except Exception:  # noqa: BLE001
            continue
        if not text or len(text) > 2000 or _looks_like_roleplay(text):
            continue
        collected.append(text)

    if not collected:
        return None
    rng.shuffle(collected)
    return collected[:n]
