"""Construct the SFT dataset (Section 4.1).

650 calm conversations (1-3 turns) mixed with 500 samples of standard instruct
data from Dolci-Instruct-SFT (to mitigate broader degeneration).  Output is a
list of conversational examples ``{"messages": [...]}`` for TRL's SFTTrainer.
"""
from __future__ import annotations

import random

from ..config import SftConfig
from .calm_data import CalmConversation


def _load_dolci(n: int, dataset_name: str, seed: int) -> list[dict]:
    """Load ``n`` instruct samples from Dolci-Instruct-SFT in conversational
    format.  Returns an empty list if the dataset is unavailable (offline)."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        examples: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                examples.append({"messages": msgs})
        return examples
    except Exception:  # noqa: BLE001 -- offline / dataset missing
        return []


def build_sft_dataset(
    calm_conversations: list[CalmConversation],
    cfg: SftConfig,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    calm = list(calm_conversations)
    rng.shuffle(calm)
    calm = calm[: cfg.n_calm]
    examples = [{"messages": c.messages} for c in calm]
    examples += _load_dolci(cfg.n_dolci, cfg.dolci_dataset, seed)
    rng.shuffle(examples)
    return examples
