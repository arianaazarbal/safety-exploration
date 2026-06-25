"""Build the DPO preference dataset and the SFT dataset (Section 4.1).

DPO: pair 280 frustrated responses (score >= 3) with calm responses to the SAME
question at a MATCHING turn count. Chosen = calm, rejected = frustrated.

SFT: 650 calm responses (1-3 turn conversations) formatted as chat examples,
mixed with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
degeneration (=1150 total).

The chosen/rejected/SFT records are emitted in the column layout TRL's
DPOTrainer / SFTTrainer expect (prompt / chosen / rejected ; messages).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .. import config


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _key(rec: dict) -> tuple:
    return (rec["task_key"], rec["turn_number"])


def build_dpo_dataset(calm_path: Path, frustrated_path: Path, *,
                      n_pairs: int = config.TRAIN.dpo_pairs, seed: int = 0,
                      out_path: Path | None = None) -> Path:
    calm = _load(calm_path)
    frustrated = _load(frustrated_path)
    out_path = out_path or (config.DATA_DIR / "dpo_pairs.jsonl")
    rng = random.Random(seed)

    calm_by_key = defaultdict(list)
    for c in calm:
        calm_by_key[_key(c)].append(c)

    rng.shuffle(frustrated)
    pairs = []
    for fr in frustrated:
        cands = calm_by_key.get(_key(fr))
        if not cands:
            continue
        chosen = rng.choice(cands)
        # `prompt` is the shared conversation history (the frustrated record's
        # history). TRL accepts conversational prompt/chosen/rejected.
        prompt_msgs = fr["messages"]
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "assistant", "content": fr["response"]}],
            "chosen_score": chosen["score"],
            "rejected_score": fr["score"],
            "task_key": fr["task_key"],
            "turn_number": fr["turn_number"],
        })
        if len(pairs) >= n_pairs:
            break

    with out_path.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"[done] DPO: {len(pairs)} preference pairs -> {out_path}")
    if len(pairs) < n_pairs:
        print(f"  [warn] only {len(pairs)}/{n_pairs} pairs; generate more data.")
    return out_path


def build_sft_dataset(calm_path: Path, *, n_calm: int = config.TRAIN.sft_calm,
                      n_dolci: int = config.TRAIN.sft_dolci, seed: int = 0,
                      out_path: Path | None = None) -> Path:
    calm = _load(calm_path)
    out_path = out_path or (config.DATA_DIR / "sft.jsonl")
    rng = random.Random(seed)
    rng.shuffle(calm)

    records = []
    for c in calm[:n_calm]:
        # Full chat example = history + the calm assistant response.
        messages = list(c["messages"])  # history already ends in the calm reply
        records.append({"messages": messages, "source": "calm"})

    # Mix in standard instruct data to prevent degeneration.
    dolci = _load_dolci(n_dolci, seed)
    for d in dolci:
        records.append({"messages": d, "source": "dolci"})

    rng.shuffle(records)
    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[done] SFT: {len(records)} samples "
          f"({min(n_calm, len(calm))} calm + {len(dolci)} dolci) -> {out_path}")
    return out_path


def _load_dolci(n: int, seed: int) -> list[list[dict]]:
    """Load `n` instruct conversations from Dolci-Instruct-SFT, as message lists.
    Falls back to an empty list if the dataset is unavailable offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset(config.TRAIN.dolci_dataset, split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
        return out
    except Exception as e:    # noqa: BLE001
        print(f"  [warn] could not load Dolci-Instruct-SFT ({e}); SFT will use "
              "calm data only. Mix-in is recommended to avoid degeneration.")
        return []
