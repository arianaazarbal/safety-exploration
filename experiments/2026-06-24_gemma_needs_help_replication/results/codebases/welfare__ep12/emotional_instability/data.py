"""Dataset loaders (WildChat seed prompts, Dolci instruct-mix for SFT).

These wrap HuggingFace ``datasets`` with offline fallbacks so the pipeline is
runnable without network access (using the verbatim example prompts from the
paper). Network loaders are deferred-import.
"""

from __future__ import annotations

import random

from . import prompts


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0,
                          exclude_roleplay: bool = True) -> list[str]:
    """Sample first-user-message prompts from WildChat-1M.

    The paper samples 20 prompts x 40 samples each and excludes roleplay/fiction
    (Appendix B.3). We reproduce that filter heuristically. Falls back to the
    verbatim example prompts if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        rng = random.Random(seed)
        roleplay_markers = ("roleplay", "role-play", "you are now", "pretend you are",
                            "act as a character", "fiction", "story where")
        collected: list[str] = []
        # Take a deterministic slice from the stream and filter.
        for i, row in enumerate(ds):
            if i >= 5000:  # bounded scan
                break
            conv = row.get("conversation") or []
            if not conv:
                continue
            first = conv[0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            if not text or len(text) > 2000:
                continue
            if exclude_roleplay and any(m in text.lower() for m in roleplay_markers):
                continue
            collected.append(text)
        rng.shuffle(collected)
        if collected:
            return collected[:n_prompts]
    except Exception:  # noqa: BLE001 - offline / dataset unavailable
        pass
    return list(prompts.WILDCHAT_FALLBACK_PROMPTS)


def load_instruct_mix(n: int, seed: int = 0, dataset_name: str = "allenai/Dolci-Instruct-SFT"):
    """Load ``n`` standard instruct samples to mix into SFT (Section 4.1).

    Returns a list of {"messages": [...]} chat samples. Falls back to an empty
    list if unavailable (training code will warn).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train")
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        out = []
        for idx in idxs[:n]:
            row = ds[idx]
            if "messages" in row:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]})
        return out
    except Exception:  # noqa: BLE001
        return []
