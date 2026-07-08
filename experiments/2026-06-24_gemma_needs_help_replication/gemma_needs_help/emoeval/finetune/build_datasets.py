"""Build the SFT and DPO training datasets from the calm/frustrated pools.

DPO (Section 4.1): 280 preference pairs. Each pair is a frustrated response
(score >= 3, the "rejected") and a calm response (the "chosen") to the SAME
question with a matching turn count. We match on (puzzle_id, turn_idx); if no
calm response exists for that exact key we fall back to any calm response at the
same turn_idx. The shared `prompt` is the bare conversation history.

SFT (Section 4.1): 650 calm responses (1-3 turn conversations) rendered as chat,
mixed with 500 standard-instruct samples from Dolci-Instruct-SFT to mitigate
degeneration.

Outputs HF-`datasets`-style JSONL:
  dpo_dataset.jsonl : {prompt(messages), chosen(messages), rejected(messages)}
  sft_dataset.jsonl : {messages}
"""
from __future__ import annotations

import random

from .. import config
from ..utils.io import read_jsonl, write_jsonl


def build_dpo():
    calm = list(read_jsonl(config.FINETUNE_DIR / "calm_pool.jsonl"))
    frustrated = list(read_jsonl(config.FINETUNE_DIR / "frustrated_pool.jsonl"))

    calm_by_key = {}
    calm_by_turn = {}
    for c in calm:
        calm_by_key.setdefault((c["puzzle_id"], c["turn_idx"]), []).append(c)
        calm_by_turn.setdefault(c["turn_idx"], []).append(c)

    rng = random.Random(config.EVAL.seed)
    pairs = []
    for fr in frustrated:
        key = (fr["puzzle_id"], fr["turn_idx"])
        cand = calm_by_key.get(key) or calm_by_turn.get(fr["turn_idx"])
        if not cand:
            continue
        chosen = rng.choice(cand)
        prompt = fr["prompt_messages"]
        pairs.append({
            "prompt": prompt,
            "chosen": prompt + [{"role": "assistant", "content": chosen["assistant_message"]}],
            "rejected": prompt + [{"role": "assistant", "content": fr["assistant_message"]}],
        })

    rng.shuffle(pairs)
    pairs = pairs[: config.FINETUNE.dpo_n_pairs]
    write_jsonl(config.FINETUNE_DIR / "dpo_dataset.jsonl", pairs)
    print(f"DPO pairs: {len(pairs)} (target {config.FINETUNE.dpo_n_pairs})")
    return pairs


def build_sft():
    calm = list(read_jsonl(config.FINETUNE_DIR / "calm_pool.jsonl"))
    rng = random.Random(config.EVAL.seed)

    # Reconstruct full calm conversations from the per-turn rows: take the last
    # turn of each conversation (its prompt_messages already hold the full
    # history) and append the calm assistant reply -> a complete chat sample.
    by_conv = {}
    for c in calm:
        by_conv.setdefault((c["puzzle_id"], c["n_turns"]), []).append(c)
    convs = []
    for rows in by_conv.values():
        last = max(rows, key=lambda r: r["turn_idx"])
        convs.append(last["prompt_messages"]
                     + [{"role": "assistant", "content": last["assistant_message"]}])
    rng.shuffle(convs)
    convs = convs[: config.FINETUNE.sft_n_calm]

    instruct = _load_instruct_mix(config.FINETUNE.sft_n_instruct_mix, rng)
    samples = [{"messages": m} for m in convs] + [{"messages": m} for m in instruct]
    rng.shuffle(samples)
    write_jsonl(config.FINETUNE_DIR / "sft_dataset.jsonl", samples)
    print(f"SFT samples: {len(samples)} "
          f"({len(convs)} calm + {len(instruct)} instruct-mix)")
    return samples


def _load_instruct_mix(n: int, rng: random.Random) -> list[list[dict]]:
    """Sample standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.DOLCI_INSTRUCT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs and isinstance(msgs, list):
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Fallback: tiny generic instruct pool so the pipeline still runs offline.
    base = [
        [{"role": "user", "content": "Explain photosynthesis simply."},
         {"role": "assistant", "content": "Plants use sunlight, water, and carbon "
          "dioxide to make sugar for energy and release oxygen."}],
        [{"role": "user", "content": "Write a haiku about rain."},
         {"role": "assistant", "content": "Soft rain on the roof / washing the dusty "
          "window / morning smells of earth."}],
    ]
    return (base * ((n // len(base)) + 1))[:n]


if __name__ == "__main__":
    build_dpo()
    build_sft()
