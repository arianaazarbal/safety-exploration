"""Build DPO preference pairs and SFT datasets from generated calm data (Sec 4.1).

DPO: pair each frustrated turn (score >= 3) with a calm response (score <= 1) to
the *same* puzzle and turn index, so chosen/rejected share an identical prompt
context. 280 pairs are sampled, biased -- as in the paper (Table 10) -- toward
mid-range frustration scores at later turns simply because those dominate the
pool.

SFT: 650 calm conversations mixed with 500 standard-instruct samples from
Dolci-Instruct-SFT to mitigate degeneration.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..utils import read_jsonl, write_jsonl


def build_dpo_pairs(cfg: Config) -> Path:
    rng = random.Random(cfg.get("seed", 0))
    data_dir = Path(cfg.get("output_dir", "runs")) / "train" / "data"
    out_path = data_dir / "dpo_pairs.jsonl"

    calm_by_key = defaultdict(list)
    for row in read_jsonl(data_dir / "calm_turns.jsonl"):
        calm_by_key[(row["puzzle_id"], row["turn_index"])].append(row)

    frustrated = [
        r
        for r in read_jsonl(data_dir / "frustrated_turns.jsonl")
        if r["score"] is not None and r["score"] >= cfg.get("training.dpo.rejected_min_score", 3)
    ]
    rng.shuffle(frustrated)

    n_pairs = cfg.get("training.dpo.n_pairs", 280)
    pairs = []
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (fr["puzzle_id"], fr["turn_index"])
        calm_candidates = calm_by_key.get(key)
        if not calm_candidates:
            continue
        calm = rng.choice(calm_candidates)
        pairs.append(
            {
                "prompt_messages": fr["prompt_messages"],
                "chosen": calm["response"],
                "rejected": fr["response"],
                "turn_index": fr["turn_index"],
                "rejected_score": fr["score"],
                "chosen_score": calm["score"],
            }
        )

    write_jsonl(out_path, pairs)
    return out_path


def build_sft_dataset(cfg: Config) -> Path:
    rng = random.Random(cfg.get("seed", 0))
    data_dir = Path(cfg.get("output_dir", "runs")) / "train" / "data"
    out_path = data_dir / "sft_dataset.jsonl"

    n_calm = cfg.get("training.sft.n_calm", 650)
    n_dolci = cfg.get("training.sft.n_dolci", 500)

    calm = list(read_jsonl(data_dir / "calm_conversations.jsonl"))
    rng.shuffle(calm)
    rows = [
        {"messages": c["messages"], "source": "calm"}
        for c in calm[:n_calm]
    ]

    rows.extend(_load_dolci(cfg, n_dolci, rng))
    rng.shuffle(rows)
    write_jsonl(out_path, rows)
    return out_path


def _load_dolci(cfg, n: int, rng) -> list[dict]:
    """Load standard instruct samples from Dolci-Instruct-SFT (Olmo 3)."""
    dataset_id = cfg.get("training.sft.dolci_dataset", "allenai/Dolci-Instruct-SFT")
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if not msgs:
                continue
            out.append({"messages": _normalize_messages(msgs), "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception:
        # Offline: return an empty list (SFT still trains on calm data alone,
        # documented as a degraded-but-runnable fallback in DESIGN.md).
        return []


def _normalize_messages(msgs):
    norm = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        norm.append({"role": role, "content": content})
    return norm
