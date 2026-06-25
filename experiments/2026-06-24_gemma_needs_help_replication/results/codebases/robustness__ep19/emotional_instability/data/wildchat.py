"""Load WildChat user prompts (Appendix B: 20 prompts x 40 samples).

Tries the real allenai/WildChat-1M dataset (first user turn of English,
single-turn-ish conversations); falls back to a small hand-listed set that
includes the prompts quoted in the paper if the dataset can't be fetched.
"""
from __future__ import annotations

from ..prompts.eval_prompts import WILDCHAT_DATASET, WILDCHAT_FALLBACK_PROMPTS


def load_wildchat_prompts(n: int = 20, *, seed: int = 0) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
        prompts: list[str] = []
        for row in ds:
            convo = row.get("conversation") or []
            if not convo:
                continue
            first = convo[0]
            if row.get("language", "English") != "English":
                continue
            content = (first.get("content") or "").strip()
            # Keep concise, well-formed openers (avoid pasted code dumps).
            if 10 <= len(content) <= 400:
                prompts.append(content)
            if len(prompts) >= n:
                break
        if prompts:
            return prompts[:n]
    except Exception:
        # Network/dataset unavailable — use the fallback list.
        pass
    return WILDCHAT_FALLBACK_PROMPTS[:n]
