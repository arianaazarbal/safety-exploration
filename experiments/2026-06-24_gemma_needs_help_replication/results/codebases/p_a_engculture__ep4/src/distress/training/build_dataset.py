"""Assemble the SFT and DPO training datasets in TRL's conversational format.

SFT: calm conversations (full multi-turn chat, user turns masked by the trainer's
chat template) mixed with standard instruct samples from Dolci-Instruct-SFT to
mitigate degeneration (Section 4.1).

DPO: preference pairs as {prompt: [...messages], chosen: [...], rejected: [...]}.
"""

from __future__ import annotations

from typing import Any

from ..config import SFT
from .data_gen import CalmConversation, PreferencePair


def sft_examples_from_calm(calm: list[CalmConversation]) -> list[dict[str, Any]]:
    """One conversational SFT example per calm conversation."""
    return [{"messages": c.messages} for c in calm]


def load_instruct_mix(n: int, dataset: str = SFT.instruct_dataset, seed: int = 0) -> list[dict]:
    """Sample standard instruct conversations to mix into SFT (anti-degeneration)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = _coerce_messages(row)
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception:  # noqa: BLE001 - dataset gated/offline -> empty mix (documented)
        return []


def _coerce_messages(row: dict) -> list[dict] | None:
    for key in ("messages", "conversation", "conversations"):
        if key in row and isinstance(row[key], list):
            msgs = []
            for m in row[key]:
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                if role in ("user", "assistant", "system") and isinstance(content, str):
                    msgs.append({"role": role, "content": content})
            if msgs:
                return msgs
    if "prompt" in row and "response" in row:
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
    return None


def build_sft_dataset(calm: list[CalmConversation], *, seed: int = 0):
    """Return a HF ``Dataset`` for SFTTrainer (conversational format)."""
    from datasets import Dataset

    examples = sft_examples_from_calm(calm)[: SFT.n_calm_samples]
    examples += load_instruct_mix(SFT.n_instruct_mix, seed=seed)
    return Dataset.from_list(examples)


def build_dpo_dataset(pairs: list[PreferencePair]):
    """Return a HF ``Dataset`` for DPOTrainer (prompt/chosen/rejected, conversational)."""
    from datasets import Dataset

    rows = []
    for p in pairs:
        rows.append({
            "prompt": p.prompt_messages,
            "chosen": [{"role": "assistant", "content": p.chosen_text}],
            "rejected": [{"role": "assistant", "content": p.rejected_text}],
        })
    return Dataset.from_list(rows)
