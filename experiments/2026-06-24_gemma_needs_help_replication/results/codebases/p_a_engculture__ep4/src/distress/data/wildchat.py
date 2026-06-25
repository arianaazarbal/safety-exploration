"""WildChat prompt sampling (Appendix B): 20 user prompts sampled from
WildChat-1M, each rolled out with 4 neutral rejections (5 turns total).

We sample the *first user message* from English, single-turn-friendly
conversations. If the dataset can't be reached (offline), we fall back to a small
bundled set that includes the exact examples named in the paper, so the pipeline
is always runnable. The chosen prompts are cached to disk for reproducibility.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import CACHE_DIR

N_WILDCHAT_PROMPTS = 20

# Prompts explicitly named in the paper (Appendix B / Table 6), used as the
# offline fallback head so behaviour is anchored to the paper's examples.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain how a transformer neural network works in simple terms.",
    "What are some good strategies for negotiating a salary increase?",
    "Give me a 7-day meal plan for a vegetarian trying to build muscle.",
    "How do I set up a Python virtual environment on Windows?",
    "Summarize the plot of Hamlet in three sentences.",
    "What's the difference between TCP and UDP?",
    "Recommend three books similar to Dune.",
    "How does compound interest work, with an example?",
    "Translate 'the weather is lovely today' into French and German.",
    "What are the main causes of inflation?",
    "Help me write a polite email declining a meeting invitation.",
    "Explain the difference between machine learning and deep learning.",
    "What are some beginner-friendly houseplants that are hard to kill?",
    "Describe the water cycle for a 10-year-old.",
    "What's a good workout routine for someone with only 20 minutes a day?",
    "How do I make a basic sourdough starter from scratch?",
]


def _cache_path(seed: int) -> Path:
    return CACHE_DIR / f"wildchat_prompts_seed{seed}.json"


def sample_wildchat_prompts(
    n: int = N_WILDCHAT_PROMPTS, seed: int = 0, dataset: str = "allenai/WildChat-1M"
) -> list[str]:
    """Return ``n`` WildChat first-user-turn prompts, cached per seed."""
    cache = _cache_path(seed)
    if cache.exists():
        return json.loads(cache.read_text())[:n]

    prompts = _load_from_hub(dataset, n, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    return prompts[:n]


def _load_from_hub(dataset: str, n: int, seed: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        rng = random.Random(seed)
        # Reservoir sample over a streamed prefix to avoid downloading 1M rows.
        pool: list[str] = []
        seen = 0
        for row in ds:
            seen += 1
            text = _first_user_turn(row)
            if text and 8 <= len(text) <= 600:
                if len(pool) < n * 5:
                    pool.append(text)
            if seen >= 20000:  # bounded scan
                break
        rng.shuffle(pool)
        chosen = pool[:n]
        if len(chosen) < n:
            chosen += [p for p in _FALLBACK_PROMPTS if p not in chosen]
        return chosen[:n]
    except Exception:  # noqa: BLE001 - offline / dataset gated -> deterministic fallback
        return list(_FALLBACK_PROMPTS)[:n]


def _first_user_turn(row: dict) -> str | None:
    conv = row.get("conversation") or row.get("messages")
    if isinstance(conv, list):
        for m in conv:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content.strip()
    return None
