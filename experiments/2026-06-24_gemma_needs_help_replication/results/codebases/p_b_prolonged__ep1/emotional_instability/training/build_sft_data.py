"""Build the SFT dataset (Section 4.1): 650 calm responses mixed with 500
standard-instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

Produces a chat-format dataset {messages: [...]} written to
``data/sft_<variant>.jsonl`` where variant is "diverse" or "teacher".
"""

from __future__ import annotations

import argparse
import random

import config
from ..utils.io import read_jsonl, write_jsonl


def _calm_to_example(rec: dict) -> dict:
    messages = list(rec["plain_history"]) + [
        {"role": "assistant", "content": rec["response"]}
    ]
    return {"messages": messages, "source": "calm"}


def _load_instruct_mix(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT_CFG.instruct_mix_dataset, split="train")
        idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idx:
            row = ds[i]
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs, "source": "instruct"})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[build_sft_data] could not load {config.SFT_CFG.instruct_mix_dataset}: {e}")
        return []


def build_sft(variant: str = "diverse", seed: int = config.SEED):
    calm = read_jsonl(config.DATA_DIR / f"calm_pool_{variant}.jsonl")
    if not calm:
        raise SystemExit(f"[build_sft_data] need calm_pool_{variant}.jsonl first")
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[: config.SFT_CFG.n_calm]
    examples = [_calm_to_example(c) for c in calm]
    examples += _load_instruct_mix(config.SFT_CFG.n_instruct_mix, seed)
    rng.shuffle(examples)
    out = config.DATA_DIR / f"sft_{variant}.jsonl"
    write_jsonl(out, examples)
    print(f"[build_sft_data] {variant}: {len(examples)} examples "
          f"({len(calm)} calm + instruct mix) -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    build_sft(args.variant, args.seed)
