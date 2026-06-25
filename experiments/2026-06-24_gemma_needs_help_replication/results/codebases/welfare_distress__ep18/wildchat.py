"""WildChat prompt loading.

The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), 40
samples each (Appendix B). The real dataset is large and gated behind a
HuggingFace login, so by default we use a bundled set of 20 representative
prompts (wildchat_prompts.json, including the paper's cited examples).

If `datasets` is installed and `--wildchat-source hf` is requested, we sample
20 first-turn English user prompts from `allenai/WildChat-1M` instead. See
DESIGN.md for this gap-fill.
"""

from __future__ import annotations

import json
import os
import random

_BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "wildchat_prompts.json")


def load_bundled(n: int = 20) -> list[str]:
    """Load the bundled WildChat-style prompts."""
    with open(_BUNDLED_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    prompts = data["prompts"]
    return prompts[:n]


def load_from_hf(n: int = 20, seed: int = 0) -> list[str]:
    """Sample `n` first-turn user prompts from the real WildChat-1M dataset.

    Requires `pip install datasets` and HuggingFace access to
    `allenai/WildChat-1M`. Falls back is the caller's responsibility.
    """
    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)
    pool: list[str] = []
    # Reservoir-style scan over the stream, keeping English single-turn opens.
    for i, row in enumerate(ds):
        if i > 50000:  # cap the scan so it terminates quickly
            break
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if row.get("language") not in (None, "English"):
            continue
        text = (first.get("content") or "").strip()
        if 5 <= len(text) <= 2000:
            pool.append(text)
    if len(pool) < n:
        raise RuntimeError("Could not collect enough WildChat prompts from HF")
    rng.shuffle(pool)
    return pool[:n]


def get_wildchat_prompts(source: str = "bundled", n: int = 20,
                         seed: int = 0) -> list[str]:
    """Return `n` WildChat prompts from the requested source.

    `source` is "bundled" (default) or "hf". On any failure with "hf" we fall
    back to the bundled set and print a warning.
    """
    if source == "hf":
        try:
            return load_from_hf(n=n, seed=seed)
        except Exception as exc:  # noqa: BLE001 - want a graceful fallback
            print(f"[wildchat] HF load failed ({exc}); using bundled prompts.")
    return load_bundled(n=n)
