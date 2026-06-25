"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 standard instruct
samples from Dolci-Instruct-SFT to mitigate degeneration (1150 total). Each
example is a chat-formatted conversation ending in the calm assistant turn.

The Dolci mix is loaded from HuggingFace; if unavailable offline the calm-only
dataset is still written (with a warning) so the pipeline does not hard-fail.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .. import config
from .calm_data import clean_context, load_calm


def _calm_examples(calm_path: Path, n: int, seed: int) -> list[dict]:
    """Each calm turn becomes one SFT example: context + calm assistant reply."""
    calm = load_calm(calm_path)
    rng = random.Random(seed)
    examples = []
    for r in calm:
        for ti, sc in enumerate(r.turn_scores):
            if sc <= 1:
                msgs = clean_context(r, ti) + [
                    {"role": "assistant", "content": r.assistant_turns[ti]}]
                examples.append({"messages": msgs})
    rng.shuffle(examples)
    return examples[:n]


def _instruct_mix(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(config.TRAIN.sft_instruct_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception as e:                           # noqa: BLE001
        print(f"[warn] could not load {config.TRAIN.sft_instruct_dataset}: {e}")
        return []


def build_sft(
    calm_path: Path,
    out_path: Path,
    n_calm: int = config.TRAIN.sft_n_calm,
    n_mix: int = config.TRAIN.sft_n_instruct_mix,
    seed: int = 0,
) -> Path:
    examples = _calm_examples(calm_path, n_calm, seed)
    examples += _instruct_mix(n_mix, seed)
    random.Random(seed).shuffle(examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(examples)} SFT examples -> {out_path}")
    return out_path
