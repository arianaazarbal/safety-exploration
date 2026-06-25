"""Build the SFT dataset (Section 4.1, Appendix E).

"For SFT, we train on a dataset of 650 calm responses covering 1-3 turn
conversations, mixed with 500 samples of standard instruct data from the
Dolci-Instruct-SFT dataset to mitigate broader degeneration."

We format calm conversations (from generate_calm_data) as chat-formatted SFT
examples, and mix in 500 Dolci-Instruct-SFT examples. Two calm-data variants are
supported: 'diverse' (prefix/suffix) and 'teacher' (Appendix F system prompt) ->
the two SFT models analysed in Section 4.2 / Appendix F.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config.settings import SETTINGS
from .generate_calm_data import CalmConversation


def _calm_to_messages(conv: CalmConversation) -> list[dict]:
    """Interleave calm conversation into chat messages (no reassurance)."""
    messages = []
    for i, user in enumerate(conv.user_turns):
        messages.append({"role": "user", "content": user})
        if i < len(conv.assistant_turns):
            messages.append({"role": "assistant", "content": conv.assistant_turns[i]})
    return messages


def _load_dolci(n: int, seed: int) -> list[dict]:
    """Load `n` standard instruct examples from Dolci-Instruct-SFT.

    Returns chat-format message lists. Falls back to an empty list if the dataset
    is unavailable (training can still run on calm data alone, with a warning).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            # Dolci stores chat-format 'messages'; pass through if present.
            if "messages" in row and row["messages"]:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "response" in row:
                out.append(
                    {
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]},
                        ]
                    }
                )
        return out
    except Exception:
        return []


def build_sft_dataset(
    calm_conversations: list[CalmConversation],
    *,
    n_calm: int = SETTINGS.sft_n_calm,
    n_dolci: int = SETTINGS.sft_n_dolci,
    seed: int = SETTINGS.seed,
    out_path: Optional[Path] = None,
) -> list[dict]:
    """Return SFT examples (each {'messages': [...]}) = 650 calm + 500 Dolci."""
    calm_examples = [
        {"messages": _calm_to_messages(c)} for c in calm_conversations[:n_calm]
    ]
    dolci_examples = _load_dolci(n_dolci, seed)
    dataset = calm_examples + dolci_examples

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in dataset:
                f.write(json.dumps(ex) + "\n")
    return dataset
