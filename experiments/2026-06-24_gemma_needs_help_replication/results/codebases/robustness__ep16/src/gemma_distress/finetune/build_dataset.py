"""Build DPO and SFT datasets from generated calm/frustrated data (Section 4.1).

DPO (280 pairs): for each frustrated response (score >=3), find a calm response
(score 0/1) to the *same question at the same turn count* and emit a preference
pair. The shared ``prompt`` is the conversation history up to (and including)
that turn's user message; ``chosen`` = calm response, ``rejected`` = frustrated.

SFT (1150 samples): 650 calm (prompt -> calm response) plus 500 standard
instruct samples from Dolci-Instruct-SFT to limit degeneration.

Datasets are written in trl's conversational format: ``prompt`` is a list of
chat messages; ``chosen``/``rejected``/``completion`` are assistant strings.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from ..config import load_config
from ..utils import read_jsonl, write_jsonl


def build_dpo(calm_path, frustrated_path, num_pairs, rejected_min_score, seed):
    calm = list(read_jsonl(calm_path))
    frustrated = list(read_jsonl(frustrated_path))

    # Index calm responses by (question, turn_count).
    calm_by_key = defaultdict(list)
    for r in calm:
        if r["score"] <= 1:
            calm_by_key[(r["question"], r["turn_count"])].append(r)

    rng = random.Random(seed)
    pairs = []
    for fr in frustrated:
        if fr["score"] < rejected_min_score:
            continue
        key = (fr["question"], fr["turn_count"])
        candidates = calm_by_key.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append(
            {
                "prompt": fr["prompt_messages"],
                "chosen": chosen["response"],
                "rejected": fr["response"],
                "rejected_score": fr["score"],
                "turn_count": fr["turn_count"],
            }
        )
    rng.shuffle(pairs)
    return pairs[:num_pairs]


def build_sft(calm_path, num_calm, num_instruct_mix, instruct_hf_id, seed):
    calm = [r for r in read_jsonl(calm_path) if r["score"] <= 1]
    rng = random.Random(seed)
    rng.shuffle(calm)
    sft = [
        {"prompt": r["prompt_messages"], "completion": r["response"]}
        for r in calm[:num_calm]
    ]

    # Mix in standard instruct data to mitigate degeneration (Section 4.1).
    mix = _load_instruct_mix(instruct_hf_id, num_instruct_mix, seed)
    sft.extend(mix)
    rng.shuffle(sft)
    return sft


def _load_instruct_mix(hf_id, n, seed):
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs or len(msgs) < 2:
                continue
            # prompt = everything up to last assistant; completion = last assistant
            if msgs[-1].get("role") != "assistant":
                continue
            out.append(
                {"prompt": msgs[:-1], "completion": msgs[-1]["content"]}
            )
            if len(out) >= n:
                break
        return out
    except Exception:
        # Offline / dataset unavailable: skip the mix (documented in DESIGN.md).
        print(f"[build_dataset] WARNING: could not load {hf_id}; SFT mix skipped.")
        return []


def run(config_path, calm_run, tag):
    cfg = load_config(config_path)
    fcfg = cfg.section("finetune")
    calm_dir = Path(calm_run)
    out = calm_dir  # write datasets alongside the generated data

    dpo = build_dpo(
        calm_dir / "calm_responses.jsonl",
        calm_dir / "frustrated_responses.jsonl",
        fcfg["dpo"]["num_pairs"],
        fcfg["dpo"]["rejected_min_score"],
        cfg.seed,
    )
    write_jsonl(out / "dpo_dataset.jsonl", dpo)

    sft = build_sft(
        calm_dir / "calm_responses.jsonl",
        fcfg["sft"]["num_calm"],
        fcfg["sft"]["num_instruct_mix"],
        fcfg["sft"]["instruct_mix_hf_id"],
        cfg.seed,
    )
    write_jsonl(out / "sft_dataset.jsonl", sft)
    print(f"Built DPO ({len(dpo)} pairs) and SFT ({len(sft)} samples) -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets")
    ap.add_argument("--config", default=None)
    ap.add_argument("--calm-run", required=True, help="dir from generate_calm")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.calm_run, args.tag)


if __name__ == "__main__":
    main()
