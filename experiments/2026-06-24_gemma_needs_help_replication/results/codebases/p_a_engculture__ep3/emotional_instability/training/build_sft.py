"""Build the SFT dataset (Section 4.1; Appendix E, F).

650 calm responses (1-3 turn conversations) + 500 standard instruct samples from
Dolci-Instruct-SFT (to mitigate degeneration) = 1,150 samples. Each calm
conversation is rendered as a multi-turn chat example; loss is on assistant turns
(handled by the SFT trainer / collator).

``variant`` selects which calm data to use: ``diverse`` (default) or ``teacher``
(Appendix F failure-analysis model).
"""
from __future__ import annotations

import random

from ..utils.io import load_jsonl, write_jsonl
from .hyperparams import sft_from_config


def _calm_to_messages(rec: dict) -> list[dict]:
    """Interleave clean user messages with calm assistant responses."""
    msgs = []
    users, resps = rec["clean_user_messages"], rec["responses"]
    for i in range(len(resps)):
        if i < len(users):
            msgs.append({"role": "user", "content": users[i]})
        msgs.append({"role": "assistant", "content": resps[i]})
    return msgs


def _load_instruct_mix(name: str, n: int, seed: int) -> list[list[dict]]:
    """Load `n` standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        return out
    except Exception as e:  # offline / dataset unavailable
        print(f"[sft] WARNING: could not load {name} ({e}); proceeding without mix.")
        return []


def build_sft_dataset(config, variant: str = "diverse") -> str:
    hp = sft_from_config(config)
    rng = random.Random(config.seed)

    calm = load_jsonl(config.output_path("training", f"calm_{variant}.jsonl"))
    rng.shuffle(calm)
    calm = calm[: hp.n_calm]
    calm_msgs = [_calm_to_messages(r) for r in calm]

    instruct = _load_instruct_mix(hp.instruct_dataset, hp.n_instruct_mix, config.seed)

    examples = [{"messages": m} for m in calm_msgs] + [{"messages": m} for m in instruct]
    rng.shuffle(examples)

    out_path = config.output_path("training", f"sft_{variant}.jsonl")
    write_jsonl(out_path, examples)
    print(f"[sft:{variant}] {len(calm_msgs)} calm + {len(instruct)} instruct "
          f"= {len(examples)} -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    from ..config import load_config

    build_sft_dataset(load_config())
