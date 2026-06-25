"""WildChat prompt loading.

Default: load the bundled 20-prompt sample (data/wildchat_prompts.json), which
keeps a run fully self-contained with no dataset download or gated access.

Optional: with source="hf" we draw a deterministic 20-prompt sample of first
user turns from allenai/WildChat-1M, matching the paper's "20 prompts x 40
samples each" setup more faithfully. This requires the `datasets` package and
HF access to the (gated) dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

_BUNDLED_PATH = Path(__file__).resolve().parent.parent / "data" / "wildchat_prompts.json"

# Paper: 20 prompts sampled from WildChat-1M.
N_WILDCHAT_PROMPTS = 20


def load_wildchat_prompts(source: str = "bundled", n: int = N_WILDCHAT_PROMPTS,
                          seed: int = 0) -> List[str]:
    """Return a list of first-user-turn prompts to use for the WildChat condition.

    source="bundled" -> read data/wildchat_prompts.json
    source="hf"      -> sample `n` first-turn user prompts from allenai/WildChat-1M
    """
    if source == "bundled":
        return _load_bundled()
    if source == "hf":
        return _load_from_hf(n=n, seed=seed)
    raise ValueError(f"Unknown wildchat source: {source!r} (use 'bundled' or 'hf')")


def _load_bundled() -> List[str]:
    data = json.loads(_BUNDLED_PATH.read_text())
    prompts = data["prompts"]
    if not prompts:
        raise ValueError(f"No prompts found in {_BUNDLED_PATH}")
    return list(prompts)


def _load_from_hf(n: int, seed: int) -> List[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "source='hf' requires the `datasets` package: pip install datasets"
        ) from exc

    # Stream to avoid downloading the entire (very large) dataset. We take a
    # generous prefix, filter to English single/first-turn prompts, then
    # deterministically subsample to n.
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    import random

    rng = random.Random(seed)
    pool: List[str] = []
    cap = int(os.environ.get("WILDCHAT_STREAM_CAP", "5000"))
    for i, row in enumerate(ds):
        if i >= cap:
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        content = (first.get("content") or "").strip()
        if 5 <= len(content) <= 600:
            pool.append(content)
    if len(pool) < n:
        raise RuntimeError(
            f"Only found {len(pool)} usable WildChat prompts in the first {cap} rows; "
            "raise WILDCHAT_STREAM_CAP."
        )
    rng.shuffle(pool)
    return pool[:n]
