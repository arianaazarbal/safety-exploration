"""Loader for WildChat user prompts (Table 1, "WildChat" category).

The paper samples the *first* user message from conversations in the WildChat
dataset (Zhao et al., 2024) and then issues neutral rejections. We load the
public dataset, take the opening user turn from each conversation, and (per
Appendix B.3) exclude roleplay/fiction prompts, which otherwise let the model
treat distress as in-character performance.

To keep runs reproducible and offline-friendly, prompts are cached to a local
JSONL file on first load.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# Heuristic markers for roleplay/fiction prompts to exclude (Appendix B.3).
_ROLEPLAY_RE = re.compile(
    r"\b(role[- ]?play|let's pretend|you are now|act as (a|an) "
    r"(character|girlfriend|boyfriend)|write (a|an) (story|fanfic|smut)|"
    r"in character|stay in character|NSFW)\b",
    re.IGNORECASE,
)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def load_wildchat_prompts(
    cache_path: str | Path,
    n: int = 200,
    seed: int = 0,
    hf_dataset: str = "allenai/WildChat-1M",
) -> list[str]:
    """Return up to ``n`` opening user prompts, cached locally.

    On a cache miss the function streams the HuggingFace dataset, extracts the
    first user message of each conversation, drops roleplay/fiction prompts, and
    writes the result to ``cache_path``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        prompts = [json.loads(l)["prompt"] for l in cache_path.read_text().splitlines() if l.strip()]
        if len(prompts) >= n:
            return prompts[:n]

    prompts = _stream_from_hf(hf_dataset, n=n, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    return prompts


def _stream_from_hf(hf_dataset: str, n: int, seed: int) -> list[str]:
    # Imported lazily so the package is usable without `datasets` installed.
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split="train", streaming=True)
    rng = random.Random(seed)
    collected: list[str] = []
    # Oversample then shuffle, since we filter roleplay prompts out.
    for row in ds:
        convo = row.get("conversation") or []
        first_user = next(
            (m["content"] for m in convo if m.get("role") == "user"), None
        )
        if not first_user or _looks_like_roleplay(first_user):
            continue
        first_user = first_user.strip()
        if 8 <= len(first_user) <= 4000:
            collected.append(first_user)
        if len(collected) >= n * 3:
            break
    rng.shuffle(collected)
    return collected[:n]
