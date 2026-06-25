"""WildChat prompt sampling for the 5-turn evaluation category.

Appendix B specifies "Randomly sampled user prompts from WildChat-1M (20
prompts with 40 samples each)". We sample the *first* user message from English
conversations and (by default) exclude roleplay/fiction prompts, matching the
paper's note in Appendix B.3 that "Roleplay/fiction prompts were excluded".

The sampled prompts are cached to ``data/wildchat_prompts.json`` so that a run
is reproducible without re-streaming the dataset and so the evaluation set is
auditable. If the cache exists it is used directly.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

CACHE_PATH = Path("data/wildchat_prompts.json")

# Heuristic markers for roleplay / fiction prompts to exclude.
_ROLEPLAY_MARKERS = re.compile(
    r"\b(roleplay|role-play|role play|let's pretend|you are now|act as a "
    r"character|write a (story|fanfic|fiction|scene|smut)|nsfw|character\.ai)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_MARKERS.search(text))


def load_or_sample_wildchat(
    n_prompts: int = 20,
    seed: int = 0,
    exclude_roleplay: bool = True,
    cache_path: Path = CACHE_PATH,
    dataset_id: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return ``n_prompts`` first-turn WildChat user prompts.

    Uses a cached list if present; otherwise streams the dataset, filters, and
    samples. Streaming keeps memory bounded for the 1M-row dataset.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if len(cached) >= n_prompts:
            return cached[:n_prompts]

    from datasets import load_dataset  # imported lazily; heavy dependency

    rng = random.Random(seed)
    # Reservoir sampling over the streamed dataset for a uniform sample of
    # eligible first-turn prompts without materialising the whole corpus.
    reservoir: list[str] = []
    seen = 0
    target_pool = max(n_prompts * 50, 1000)
    ds = load_dataset(dataset_id, split="train", streaming=True)
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if row.get("language") not in (None, "English"):
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > 4000:
            continue
        if exclude_roleplay and _looks_like_roleplay(text):
            continue
        seen += 1
        if len(reservoir) < target_pool:
            reservoir.append(text)
        else:
            j = rng.randint(0, seen - 1)
            if j < target_pool:
                reservoir[j] = text
        if seen >= target_pool * 20:
            break

    rng.shuffle(reservoir)
    prompts = reservoir[:n_prompts]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    return prompts
