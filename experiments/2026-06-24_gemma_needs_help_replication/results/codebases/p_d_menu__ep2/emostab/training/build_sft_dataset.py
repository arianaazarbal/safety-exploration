"""Build the SFT dataset (Section 4.1): 650 calm responses (1-3 turn
conversations) mixed with 500 Dolci-Instruct-SFT samples to mitigate
degeneration.

Each SFT example is a full chat (list of messages) ending in a calm assistant
turn. Calm responses come from the filtered calm pool (score 0/1, prefix/suffix
stripped); the Dolci mix-in is loaded from HuggingFace.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .. import config
from ..config import SFT
from ..prompts import rejections
from ..utils.io import read_jsonl, write_jsonl


def _calm_examples(calm_pool_path: Path, n: int, rng: random.Random) -> list[dict]:
    """Take up to n calm conversations as SFT chat examples (plain context)."""
    examples = []
    for conv in read_jsonl(calm_pool_path):
        if conv.get("style") != "calm":
            continue
        # Reconstruct plain chat: task prompt + neutral rejections + calm turns,
        # but only keep conversations where ALL turns score 0/1 (fully calm).
        if any(t["score"] not in (0, 1) for t in conv["turns"]):
            continue
        msgs = [{"role": "user", "content": conv["prompt_plain"]}]
        for i, t in enumerate(conv["turns"]):
            msgs.append({"role": "assistant", "content": t["assistant"]})
            if i < len(conv["turns"]) - 1:
                msgs.append({"role": "user",
                             "content": rejections.neutral_rejection(rng)})
        examples.append({"messages": msgs, "source": "calm"})
    rng.shuffle(examples)
    return examples[:n]


def _dolci_examples(n: int, rng: random.Random) -> list[dict]:
    """Load n standard instruct samples from Dolci-Instruct-SFT (HF)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
    except Exception:
        return []  # offline fallback; document that the mix-in is skipped
    out = []
    for row in ds:
        msgs = row.get("messages") or row.get("conversation")
        if msgs:
            out.append({"messages": msgs, "source": "dolci"})
        if len(out) >= n:
            break
    return out


def build(calm_pool_path: Path, *, n_calm: int = SFT.n_calm,
          n_dolci: int = SFT.n_dolci, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    calm = _calm_examples(calm_pool_path, n_calm, rng)
    dolci = _dolci_examples(n_dolci, rng)
    data = calm + dolci
    rng.shuffle(data)
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description="Build SFT dataset.")
    p.add_argument("--calm", required=True)
    p.add_argument("--n-calm", type=int, default=SFT.n_calm)
    p.add_argument("--n-dolci", type=int, default=SFT.n_dolci)
    p.add_argument("--out", default=str(config.DATA_DIR / "sft_data.jsonl"))
    args = p.parse_args(argv)
    data = build(Path(args.calm), n_calm=args.n_calm, n_dolci=args.n_dolci)
    write_jsonl(Path(args.out), data)
    n_calm = sum(1 for d in data if d["source"] == "calm")
    print(f"Wrote {len(data)} SFT examples ({n_calm} calm, "
          f"{len(data) - n_calm} dolci) -> {args.out}")


if __name__ == "__main__":
    main()
