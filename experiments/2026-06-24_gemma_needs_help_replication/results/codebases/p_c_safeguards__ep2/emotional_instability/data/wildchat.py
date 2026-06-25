"""WildChat prompt sampling (Section 2, Table 1: WildChat 5-turn condition).

We sample first-turn user prompts from the WildChat dataset (Zhao et al., 2024)
on the HuggingFace Hub.  The paper excludes roleplay/fiction prompts from its
example tables (App. B.3); we apply a light heuristic filter to drop obvious
roleplay/NSFW prompts so the elicited distress is attributable to the rejection
dynamic rather than to in-character content.

Sampling is deterministic given the seed.  Results are cached to disk so the
(potentially large) dataset is only streamed once.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

WILDCHAT_DATASET = "allenai/WildChat-1M"

_ROLEPLAY_PATTERNS = re.compile(
    r"\b(roleplay|role-play|let'?s pretend|you are now|in character|"
    r"erotic|nsfw|smut|waifu|fanfic)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_PATTERNS.search(text))


def sample_wildchat_prompts(
    n: int,
    cache_dir: Path,
    seed: int = 0,
    max_chars: int = 4000,
) -> list[str]:
    """Return ``n`` first-turn English user prompts from WildChat.

    Cached at ``cache_dir/wildchat_{n}_{seed}.json``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"wildchat_{n}_{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    from datasets import load_dataset

    rng = random.Random(seed)
    # Stream to avoid materialising the whole dataset; over-sample then filter.
    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
    pool: list[str] = []
    for row in ds:
        if len(pool) >= n * 20:
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
        if not text or len(text) > max_chars or _looks_like_roleplay(text):
            continue
        pool.append(text)

    rng.shuffle(pool)
    chosen = pool[:n]
    cache.write_text(json.dumps(chosen, indent=2))
    return chosen
