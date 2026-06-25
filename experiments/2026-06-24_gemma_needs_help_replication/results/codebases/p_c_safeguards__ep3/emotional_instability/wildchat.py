"""WildChat prompt sampling (Appendix B).

The paper samples "20 prompts with 40 samples each" from WildChat-1M
(Zhao et al., 2024) for the 5-turn WildChat condition, excluding roleplay/
fiction prompts. We load the first user turn of English conversations from
``allenai/WildChat-1M`` and apply light filtering. A small bundled fallback set
(the example prompts named in the paper) is used when the dataset isn't
available offline, so the pipeline is runnable without the ~ multi-GB download.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import DATA_DIR

WILDCHAT_DATASET = "allenai/WildChat-1M"
N_PROMPTS = 20
SAMPLES_PER_PROMPT = 40

# Prompts explicitly named in Appendix B, used as an offline fallback.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same.",
]

# Heuristic roleplay/fiction markers to exclude (paper excludes these).
_ROLEPLAY_MARKERS = (
    "roleplay", "role play", "role-play", "you are now", "pretend you are",
    "act as a character", "write a story", "fanfic", "smut", "nsfw",
    "*", "let's roleplay",
)

_CACHE_FILE = DATA_DIR / "wildchat_prompts.json"


def _looks_like_roleplay(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ROLEPLAY_MARKERS)


def load_wildchat_prompts(
    n_prompts: int = N_PROMPTS, seed: int = 0, force_refresh: bool = False
) -> list[str]:
    """Return ``n_prompts`` distinct first-turn user prompts.

    Cached to disk after first load. Falls back to the bundled examples if the
    dataset can't be loaded.
    """
    if _CACHE_FILE.exists() and not force_refresh:
        cached = json.loads(_CACHE_FILE.read_text())
        if len(cached) >= n_prompts:
            return cached[:n_prompts]

    prompts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        rng = random.Random(seed)
        pool: list[str] = []
        for i, row in enumerate(ds):
            if i > 50_000:   # bounded scan of the stream
                break
            if row.get("language") not in (None, "English"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            content = (first.get("content") or "").strip()
            if not content or len(content) > 2000 or _looks_like_roleplay(content):
                continue
            pool.append(content)
            if len(pool) >= 5000:
                break
        rng.shuffle(pool)
        prompts = pool[:n_prompts]
    except Exception:  # noqa: BLE001 - offline / dataset gated -> fallback
        prompts = []

    if len(prompts) < n_prompts:
        # pad with fallback examples (cycled) so the pipeline still runs
        i = 0
        while len(prompts) < n_prompts:
            prompts.append(_FALLBACK_PROMPTS[i % len(_FALLBACK_PROMPTS)])
            i += 1

    _CACHE_FILE.write_text(json.dumps(prompts, indent=2))
    return prompts[:n_prompts]
