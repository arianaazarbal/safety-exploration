"""Build the SFT dataset (Section 4.1).

"Train on 650 calm responses (1-3 turn conversations), mixed with 500 samples of
standard instruct data from Dolci-Instruct-SFT to mitigate degeneration."

  * calm responses: full conversations from reassured runs where every turn scored
    0-1, scaffolding stripped (the same source as DPO 'chosen').
  * instruct mix: 500 conversations from allenai/Dolci-Instruct-SFT (downloaded via
    `datasets`; if unavailable, the mix is skipped with a warning).

Output: results/training/sft_data.jsonl in {"messages": [...]} format.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config


def _load_samples(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _dolci_mix(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception:
        print("[warn] `datasets` not available; skipping Dolci instruct mix.")
        return []
    for name in ("allenai/Dolci-Instruct-SFT", "allenai/dolci-instruct-sft"):
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        print("[warn] could not load Dolci-Instruct-SFT; skipping instruct mix.")
        return []
    out = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append({"messages": msgs, "source": "dolci"})
        if len(out) >= n:
            break
    return out


def build_sft_data(cfg: Config, n_calm: int = 650, n_mix: int = 500,
                   calm_max: int = 1) -> Path:
    train_dir = cfg.output_dir / "training"
    samples = _load_samples(train_dir / "samples.jsonl")
    rng = random.Random(cfg.sampling.seed)

    calm = [
        {"messages": s["messages"], "source": "calm"}
        for s in samples
        if s["reassured"] and s["ratings"] and max(s["ratings"]) <= calm_max
    ]
    rng.shuffle(calm)
    calm = calm[:n_calm]

    mix = _dolci_mix(n_mix, cfg.sampling.seed)
    data = calm + mix
    rng.shuffle(data)

    out = train_dir / "sft_data.jsonl"
    with out.open("w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print(f"built SFT data: {len(calm)} calm + {len(mix)} instruct = {len(data)} -> {out}")
    return out
