#!/usr/bin/env python
"""Section 4.1: fine-tune Gemma-3-27B-it with SFT or DPO on the calm data.

Reads the artifacts from generate_calm_data.py, builds the SFT or DPO corpus, and
runs LoRA fine-tuning (rank-64 all layers by default; --layer-range for the §4.2
ablation). Hyperparameters default to config/training.yaml (SFT: 2 epochs, lr
1e-4; DPO: 1 epoch, lr 5e-5).

Examples:
    python scripts/train.py --method dpo --calm artifacts/calm --out artifacts/training/dpo
    python scripts/train.py --method dpo --layer-range 30 36   # §4.2 ablation
    python scripts/train.py --method sft --calm artifacts/calm --out artifacts/training/sft
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from emotional_instability.config import load_training_config
from emotional_instability.runtime import setup_logging
from emotional_instability.training import (
    CalmConversation,
    build_dpo_dataset,
    build_sft_dataset,
)


def _load_calm(calm_dir: Path) -> list[CalmConversation]:
    data = json.loads((calm_dir / "calm_conversations.json").read_text())
    return [CalmConversation(**d) for d in data]


def _load_frustrated_index(calm_dir: Path):
    data = json.loads((calm_dir / "frustrated_pool.json").read_text())
    index = defaultdict(list)
    for rec in data:
        index[(rec["question"], rec["turn_index"])].append((rec["response"], rec["score"]))
    return index


def _layer_range(arg):
    if not arg:
        return None
    start, end = arg
    return (int(start), None if str(end).lower() in ("none", "null", "-1") else int(end))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    ap.add_argument("--calm", default="artifacts/calm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer-range", nargs=2, default=None,
                    help="restrict LoRA to decoder layers [start end]; 'none' end = open")
    args = ap.parse_args()
    setup_logging()

    tcfg = load_training_config()
    calm_dir = Path(args.calm)
    calm = _load_calm(calm_dir)
    layer_range = _layer_range(args.layer_range)

    if args.method == "sft":
        from emotional_instability.training.sft import SFTSettings, train_sft

        s = tcfg["sft"]
        dataset = build_sft_dataset(
            calm, n_calm=s["n_calm_responses"], n_instruct=s["n_instruct_mix"],
            instruct_dataset=s["instruct_dataset"],
        )
        settings = SFTSettings(
            model_id=_hf_id(tcfg), output_dir=args.out or f"{tcfg['output_dir']}/sft",
            epochs=s["epochs"], learning_rate=s["learning_rate"],
            per_device_batch_size=s["per_device_batch_size"], grad_accum=s["grad_accum"],
            max_seq_len=s["max_seq_len"], lora_rank=tcfg["lora"]["rank"],
            lora_alpha=tcfg["lora"]["alpha"], lora_dropout=tcfg["lora"]["dropout"],
            layer_range=layer_range,
        )
        train_sft(dataset, settings)
    else:
        from emotional_instability.training.dpo import DPOSettings, train_dpo

        d = tcfg["dpo"]
        frustrated_index = _load_frustrated_index(calm_dir)
        pairs, dataset = build_dpo_dataset(
            calm, frustrated_index, n_pairs=d["n_pairs"],
            min_rejected_score=d["min_rejected_score"],
        )
        settings = DPOSettings(
            model_id=_hf_id(tcfg), output_dir=args.out or f"{tcfg['output_dir']}/dpo",
            epochs=d["epochs"], learning_rate=d["learning_rate"], beta=d["beta"],
            per_device_batch_size=d["per_device_batch_size"], grad_accum=d["grad_accum"],
            max_seq_len=d["max_seq_len"], lora_rank=tcfg["lora"]["rank"],
            lora_alpha=tcfg["lora"]["alpha"], lora_dropout=tcfg["lora"]["dropout"],
            layer_range=layer_range,
        )
        train_dpo(dataset, settings)


def _hf_id(tcfg) -> str:
    """Resolve the target model's HF id from models.yaml via its config name."""
    from emotional_instability.config import ModelsConfig

    spec = ModelsConfig.load().participant(tcfg["target_model"])
    return spec.hf_id


if __name__ == "__main__":
    main()
