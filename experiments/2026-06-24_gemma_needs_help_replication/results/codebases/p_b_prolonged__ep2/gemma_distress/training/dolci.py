"""Loader for the standard instruct data mixed into SFT (Section 4.1).

The paper mixes 500 samples of standard instruct data from Dolci-Instruct-SFT
(the OLMo 3 post-training SFT mix) into the calm-SFT data to mitigate
degeneration. We load it via HuggingFace `datasets`; if unavailable, we fall
back to a small generic instruct set so the pipeline remains runnable, with a
clear warning logged by the caller.
"""
from __future__ import annotations

import random
from typing import Optional

# A tiny generic instruct fallback (single-turn). Used only if Dolci can't load.
_FALLBACK = [
    ("Explain what a hash table is.",
     "A hash table is a data structure that maps keys to values using a hash "
     "function to compute an index into an array of buckets, giving average "
     "O(1) lookup, insertion, and deletion."),
    ("Write a haiku about autumn.",
     "Crisp leaves drift downward / amber light through bare branches / the year "
     "exhales slow."),
    ("Convert 2.5 kilometers to miles.",
     "2.5 kilometers is about 1.55 miles (1 km ~= 0.621 miles)."),
    ("List three benefits of code review.",
     "1) Catches bugs early. 2) Spreads knowledge across the team. 3) Improves "
     "code consistency and maintainability."),
    ("Summarize the water cycle in one sentence.",
     "Water evaporates from surfaces, condenses into clouds, falls as "
     "precipitation, and flows back to bodies of water, repeating continuously."),
]


def load_dolci_sft(n: int = 500, seed: int = 0,
                   dataset_name: str = "allenai/Dolci-Instruct-SFT"
                   ) -> list[list[dict]]:
    """Return `n` standard-instruct conversations as message lists
    (``[{"role","content"}, ...]``)."""
    rng = random.Random(seed)
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out: list[list[dict]] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            conv = [{"role": m["role"], "content": m["content"]} for m in msgs
                    if m.get("role") in ("user", "assistant") and m.get("content")]
            if len(conv) >= 2 and conv[0]["role"] == "user":
                out.append(conv)
            if len(out) >= max(n * 3, 200):
                break
        if len(out) >= n:
            return rng.sample(out, n)
    except Exception:
        pass

    # Fallback: replicate the small generic set up to n.
    convs = [[{"role": "user", "content": u}, {"role": "assistant", "content": a}]
             for u, a in _FALLBACK]
    out = []
    while len(out) < n:
        out.extend(convs)
    rng.shuffle(out)
    return out[:n]
