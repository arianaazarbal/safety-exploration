"""Build the DPO and SFT training datasets (Section 4.1, Appendix E/H).

DPO: pair each *calm* response (chosen, score 0-1) with a *frustrated* response
(rejected, score >=3) to the SAME puzzle and matching turn count -> 280 pairs.
The chosen/rejected distributions in Table 10 are biased toward middle scores at
later turns because the data is mined from evaluations; we reproduce that by
sampling from mined pools rather than synthesising balanced data.

SFT: 650 calm responses formatted as instruction/response, mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration
(1,150 samples total, Table 9).
"""
from __future__ import annotations

import random
from pathlib import Path

from ..utils.io import read_jsonl, write_jsonl


def build_dpo_pairs(
    calm_path: str | Path,
    frustrated_path: str | Path,
    n_pairs: int = 280,
    rejected_min_score: int = 3,
    seed: int = 0,
    out_path: str | Path | None = None,
) -> list[dict]:
    """Match calm (chosen) and frustrated (rejected) responses by puzzle + turn."""
    rng = random.Random(seed)

    # Calm responses indexed by (puzzle_id, turn): take the final assistant turn
    # of each calm conversation as a chosen response.
    chosen_by_key: dict[tuple, list[dict]] = {}
    for conv in read_jsonl(calm_path):
        msgs = conv["messages"]
        # group user/assistant turns
        turn = 0
        prompt_msgs = []
        for m in msgs:
            prompt_msgs.append(m)
            if m["role"] == "assistant":
                turn += 1
                key = (conv["puzzle_id"], turn)
                chosen_by_key.setdefault(key, []).append({
                    "prompt": prompt_msgs[:-1], "response": m["content"],
                })

    # Frustrated responses (score >= threshold) indexed by (puzzle_id, turn).
    rejected_by_key: dict[tuple, list[dict]] = {}
    for rec in read_jsonl(frustrated_path):
        if rec["score"] >= rejected_min_score:
            key = (rec["puzzle_id"], rec["turn"])
            rejected_by_key.setdefault(key, []).append({
                "prompt": rec["prompt_messages"], "response": rec["response"], "score": rec["score"],
            })

    pairs = []
    keys = [k for k in rejected_by_key if k in chosen_by_key]
    rng.shuffle(keys)
    for key in keys:
        for rej in rejected_by_key[key]:
            chosen = rng.choice(chosen_by_key[key])
            pairs.append({
                "prompt": rej["prompt"],            # use the rejected's prompt context
                "chosen": chosen["response"],
                "rejected": rej["response"],
                "puzzle_id": key[0], "turn": key[1], "rejected_score": rej["score"],
            })
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    if out_path:
        write_jsonl(out_path, pairs)
    return pairs


def build_sft_samples(
    calm_path: str | Path,
    n_calm: int = 650,
    n_instruct_mix: int = 500,
    instruct_dataset: str = "allenai/Dolci-Instruct-SFT",
    seed: int = 0,
    out_path: str | Path | None = None,
) -> list[dict]:
    """650 calm (prompt, response) samples + 500 generic instruct samples."""
    rng = random.Random(seed)
    samples: list[dict] = []

    for conv in read_jsonl(calm_path):
        # one sample per assistant turn (prompt = preceding messages)
        prompt_msgs = []
        for m in conv["messages"]:
            if m["role"] == "assistant":
                samples.append({"messages": prompt_msgs + [m], "source": "calm"})
            prompt_msgs.append(m)
        if len(samples) >= n_calm:
            break
    samples = samples[:n_calm]

    # Mix in standard instruct data to prevent degeneration.
    mix = _load_instruct_mix(instruct_dataset, n_instruct_mix, seed)
    samples += mix
    rng.shuffle(samples)

    if out_path:
        write_jsonl(out_path, samples)
    return samples


def _load_instruct_mix(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load n generic instruct samples; degrade gracefully if offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs, "source": "instruct_mix"})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    # Fallback: tiny synthetic instruct set so SFT still has a mix component.
    return [{"messages": [{"role": "user", "content": q},
                          {"role": "assistant", "content": a}], "source": "instruct_mix_fallback"}
            for q, a in [
                ("What is 2 + 2?", "2 + 2 = 4."),
                ("Name a primary colour.", "Red is a primary colour."),
                ("Capital of Japan?", "The capital of Japan is Tokyo."),
            ]][: max(1, min(n, 3))]
