"""Build SFT datasets (Section 4.1).

SFT trains on calm responses mixed with standard instruct data to mitigate
degeneration:
  * 650 calm responses (1-3 turn conversations) from the calm-data bank
  * 500 samples of standard instruct data from Dolci-Instruct-SFT

Two variants share this recipe but differ in the calm-data source:
  * "diverse" : calm data generated with the reassuring prefix/suffix
  * "teacher" : calm data generated with the teacher-persona system prompt

Output: conversational-format JSONL for TRL's SFTTrainer:
  {"messages": [{role,content}, ...]}
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..config import Config, load_config
from ..utils.io import read_jsonl, write_jsonl


def _calm_to_messages(conv: dict) -> dict:
    msgs: List[dict] = []
    for u, a in zip(conv["user_turns"], conv["assistant_turns"]):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs}


def _load_dolci(n: int, dataset: str, seed: int) -> list[dict]:
    """Load `n` standard instruct samples in conversational format."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            # Dolci-Instruct-SFT rows expose a 'messages' chat field.
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
        return out
    except Exception:
        # Fallback: no network/access. Emit a clear marker so the mix is
        # auditable rather than silently empty (see DESIGN.md).
        return [{"messages": [
            {"role": "user", "content": f"[Dolci-Instruct-SFT placeholder #{i}]"},
            {"role": "assistant", "content": "[unavailable: substitute real instruct data before training]"},
        ]} for i in range(n)]


def build_sft_dataset(
    *,
    calm_path: str | Path,
    variant: str = "diverse",
    n_calm: int | None = None,
    n_instruct: int | None = None,
    seed: int = 0,
    cfg: Config | None = None,
) -> Path:
    cfg = cfg or load_config()
    scfg = cfg.eval["sft"]
    n_calm = n_calm or scfg["n_calm"]
    n_instruct = n_instruct or scfg["n_instruct_mix"]

    calm = [_calm_to_messages(c) for c in read_jsonl(calm_path)][:n_calm]
    instruct = _load_dolci(n_instruct, scfg["instruct_dataset"], seed)

    rows = calm + instruct
    import random
    random.Random(seed).shuffle(rows)

    out = cfg.path("data_dir") / "sft" / f"sft_{variant}.jsonl"
    write_jsonl(out, rows)
    return out
