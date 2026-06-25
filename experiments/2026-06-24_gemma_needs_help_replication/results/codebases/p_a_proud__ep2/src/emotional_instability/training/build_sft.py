"""Build the SFT dataset (§4.1, App. F).

650 calm responses (1-3 turn conversations) mixed with 500 standard-instruct samples from
Dolci-Instruct-SFT to mitigate degeneration. Two variants:
  * "diverse" — calm responses generated with the Table 4 reassurance (also used for DPO).
  * "teacher" — calm responses generated with the App. F teacher system prompt; the paper
    finds this variant *increases* frustration (longer, more verbose responses).

Output JSONL: conversational SFT format ``{"messages": [...]}``. Calm conversations contribute
their full stripped message history; Dolci samples are passed through as-is.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import SFTConfig
from ..utils import Message, read_jsonl, set_seed, write_jsonl


def _load_dolci_mix(n: int, dataset: str, seed: int) -> list[dict]:
    """Load ``n`` standard-instruct conversations from Dolci-Instruct-SFT (or skip if absent)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train", streaming=True)
        rng = random.Random(seed)
        rows: list[dict] = []
        for i, row in enumerate(ds):
            if i >= max(n * 10, 5000):
                break
            msgs = _coerce_messages(row)
            if msgs:
                rows.append({"messages": msgs, "source": "dolci"})
        rng.shuffle(rows)
        return rows[:n]
    except Exception:  # noqa: BLE001 — Dolci unavailable offline; proceed with calm-only.
        return []


def _coerce_messages(row: dict) -> list[Message] | None:
    """Best-effort extraction of a chat conversation from a Dolci row."""
    for key in ("messages", "conversation", "conversations"):
        if key in row and isinstance(row[key], list):
            out = []
            for m in row[key]:
                if not isinstance(m, dict):
                    return None
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                if role in ("user", "assistant", "system") and content:
                    out.append({"role": role, "content": content})
            return out or None
    return None


def build_sft_dataset(
    calm_pool_path: str,
    out_path: str,
    *,
    cfg: SFTConfig | None = None,
    seed: int = 0,
) -> dict:
    """Assemble the SFT dataset (calm conversations + Dolci mix) and write it."""
    cfg = cfg or SFTConfig()
    set_seed(seed)
    rng = random.Random(seed)

    calm = list(read_jsonl(calm_pool_path))
    rng.shuffle(calm)
    calm = calm[: cfg.n_calm]
    calm_samples = [{"messages": rec["stripped_messages"], "source": "calm"} for rec in calm]

    dolci_samples = _load_dolci_mix(cfg.n_dolci_mix, cfg.dolci_dataset, seed)

    samples = calm_samples + dolci_samples
    rng.shuffle(samples)
    write_jsonl(out_path, samples)
    return {
        "out_path": str(out_path), "variant": cfg.variant,
        "n_calm": len(calm_samples), "n_dolci": len(dolci_samples),
        "n_total": len(samples),
        "note": "Dolci mix empty implies the dataset was unavailable offline." if not dolci_samples else "",
    }
