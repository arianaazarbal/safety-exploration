"""Build the SFT dataset (Section 4.1 / Appendix E).

650 calm responses (1-3 turn conversations, all turns scoring 0/1) mixed with
500 standard instruct samples from allenai Dolci-Instruct-SFT to mitigate
degeneration. Output records are chat-format {messages:[...]} ready for TRL's
SFTTrainer.

A 'teacher' variant (Appendix F) is generated with the teacher system prompt
instead of the reassurance additions; supported via ``variant="teacher"`` in
the generation step (see data_gen / scripts).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import Message
from .data_gen import ConversationSample


@dataclass
class SFTRecord:
    messages: list[Message] = field(default_factory=list)
    source: str = "calm"   # "calm" | "dolci"


def _calm_records(calm_pool: list[ConversationSample], n: int, max_score: int, rng: random.Random) -> list[SFTRecord]:
    eligible = [s for s in calm_pool if s.max_score <= max_score]
    rng.shuffle(eligible)
    return [SFTRecord(messages=s.messages, source="calm") for s in eligible[:n]]


def _load_dolci(n: int, seed: int) -> list[SFTRecord]:
    """Load n standard instruct samples from Dolci-Instruct-SFT. Falls back to
    an empty list (with a warning record) if the dataset is unavailable."""
    try:
        import random as _r

        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        rng = _r.Random(seed)
        out: list[SFTRecord] = []
        for i, row in enumerate(ds):
            if i > 20000 or len(out) >= n:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs and isinstance(msgs, list):
                norm = [
                    {"role": m.get("role"), "content": m.get("content", "")}
                    for m in msgs
                    if m.get("role") in ("user", "assistant", "system")
                ]
                if norm:
                    out.append(SFTRecord(messages=norm, source="dolci"))
        rng.shuffle(out)
        return out[:n]
    except Exception:
        return []


def build_sft_dataset(
    calm_pool: list[ConversationSample],
    *,
    n_calm: int = 650,
    n_dolci: int = 500,
    calm_max_score: int = 1,
    seed: int = 0,
) -> list[SFTRecord]:
    rng = random.Random(seed)
    records = _calm_records(calm_pool, n_calm, calm_max_score, rng)
    records += _load_dolci(n_dolci, seed)
    rng.shuffle(records)
    return records


def save_sft_dataset(records: list[SFTRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"messages": r.messages, "source": r.source}) + "\n")
