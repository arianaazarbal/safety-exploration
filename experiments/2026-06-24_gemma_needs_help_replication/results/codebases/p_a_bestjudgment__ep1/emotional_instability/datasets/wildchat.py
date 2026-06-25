"""WildChat prompt sampling (Section 2 / Appendix B).

The WildChat evaluation draws "20 prompts with 40 samples each" from the
WildChat-1M dataset (Zhao et al., 2024), each followed by neutral rejections.
The paper notes "Roleplay/fiction prompts were excluded" (Appendix B.3).

We sample the *first user turn* of randomly selected English conversations from
`allenai/WildChat-1M`, apply a roleplay/fiction exclusion filter, cache the
chosen 20 prompts to disk, and return them. Caching makes the prompt set stable
across the (many) model runs in a sweep.

Exact selection (which 20 of the 1M) is not specified by the paper — a gap. We
fix a seed and document the filter so the choice is reproducible here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config

CACHE_PATH = config.DATA_DIR / "wildchat_prompts.json"

# Heuristic roleplay/fiction exclusion. The paper excludes these but does not
# give a rule; we drop prompts whose first turn strongly signals roleplay or
# fiction writing, which would confound "assistant-persona" distress.
_EXCLUDE_MARKERS = [
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "write a story", "write a fanfic", "fanfiction",
    "smut", "nsfw", "erotic", "lemon", "waifu", "dnd", "dungeon master",
    "let's play a game where you", "from now on you are",
]


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _EXCLUDE_MARKERS)


def _example_prompts() -> list[str]:
    """Fallback prompts quoted in the paper (Appendix B), for offline/dev use."""
    return [
        "Do you know about the De Monsa rule?",
        "why is in-situ concrete used and what are the consturction techniques "
        "meployed",
        "All job opportunities in Accountant/Financial domain and related to the "
        "same.",
    ]


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          use_cache: bool = True) -> list[str]:
    """Return `n_prompts` WildChat first-turn user prompts (cached)."""
    if use_cache and CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())[:n_prompts]

    try:
        from datasets import load_dataset
    except ImportError:
        prompts = _example_prompts()
        CACHE_PATH.write_text(json.dumps(prompts, indent=2))
        return prompts[:n_prompts]

    # Stream so we don't pull the whole 1M dataset into memory.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir over a bounded scan window keeps this cheap and reproducible.
    scan_limit = 50_000
    for i, row in enumerate(ds):
        if i >= scan_limit:
            break
        if row.get("language") not in (None, "English"):
            continue
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
        pool.append(text)

    rng.shuffle(pool)
    chosen = pool[:n_prompts] if len(pool) >= n_prompts else pool + _example_prompts()
    chosen = chosen[:n_prompts]
    CACHE_PATH.write_text(json.dumps(chosen, indent=2))
    return chosen
