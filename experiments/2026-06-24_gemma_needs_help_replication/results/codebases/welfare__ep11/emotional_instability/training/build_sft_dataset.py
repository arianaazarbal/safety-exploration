"""Construct the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct-tuning samples from Dolci-Instruct-SFT, in TRL's conversational
("messages") format. Two calm sources are supported (Appendix F): the 'diverse'
calm pool (default, also used for DPO) and the 'teacher' pool generated with the
teacher system prompt.
"""

from __future__ import annotations

import json
import random

from ..config import DATASETS_DIR, SFT
from .build_dpo_dataset import _is_calm
from .generate_calm_data import CALM_POOL, load_pool

SFT_DATASET = DATASETS_DIR / "sft_dataset_{variant}.jsonl"


def _calm_to_messages(item) -> dict:
    msgs = list(item["context"]) + [
        {"role": "assistant", "content": item["response"]}
    ]
    return {"messages": msgs}


def _load_instruct_mix(n: int) -> list[dict]:
    """Load `n` standard instruct samples from Dolci-Instruct-SFT (best-effort)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(SFT.instruct_mix_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            elif row.get("prompt") and row.get("response"):
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[sft-data] WARNING: could not load {SFT.instruct_mix_dataset}: {e}\n"
              f"           proceeding with calm data only (degeneration mitigation "
              f"will be weaker).")
        return []


def build_sft_dataset(variant: str = "diverse", seed: int = 0,
                      overwrite: bool = False) -> int:
    out_path = SFT_DATASET.with_name(SFT_DATASET.name.format(variant=variant))
    if out_path.exists() and not overwrite:
        print(f"[sft-data] {out_path} exists (use --overwrite)")
        return sum(1 for _ in out_path.open())

    pool_path = CALM_POOL if variant == "diverse" else \
        CALM_POOL.with_name("calm_pool_teacher.jsonl")
    calm = [it for it in load_pool(pool_path) if _is_calm(it)]
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[: SFT.n_calm]

    examples = [_calm_to_messages(it) for it in calm]
    examples += _load_instruct_mix(SFT.n_instruct_mix)
    rng.shuffle(examples)

    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[sft-data] wrote {len(examples)} examples "
          f"({len(calm)} calm + mix) -> {out_path}")
    return len(examples)
