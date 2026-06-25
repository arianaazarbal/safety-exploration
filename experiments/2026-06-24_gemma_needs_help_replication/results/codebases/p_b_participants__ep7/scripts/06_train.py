#!/usr/bin/env python3
"""Section 4.1: build DPO/SFT datasets and (optionally) run LoRA finetuning.

Dataset construction uses only already-collected data (no new distress). Pass
--train to actually launch finetuning (requires the torch/peft/trl stack and a
GPU). Adapters are written to outputs/training/{dpo,sft}; point the
gemma-3-27b-it-{dpo,sft} model keys' adapter_path at them to evaluate.
"""
from __future__ import annotations

from _common import base_parser, load

from distress_eval.io_utils import read_jsonl, write_jsonl
from distress_eval.training import (
    ConvSample,
    build_dpo_pairs,
    rollouts_to_samples,
)
from distress_eval.training.datasets import build_sft_dataset


def _load_conv_samples(path):
    return [ConvSample(**d) for d in read_jsonl(path)]


def main():
    p = base_parser(__doc__)
    p.add_argument("--method", choices=["dpo", "sft", "both"], default="dpo")
    p.add_argument("--source-model", default="gemma-3-27b-it",
                   help="Model providing frustrated (rejected) responses for DPO")
    p.add_argument("--train", action="store_true", help="Actually run finetuning")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Restrict LoRA to these layer indices (§4.2 ablation)")
    args = p.parse_args()
    cfg = load(args)
    if args.layers:
        cfg.training.layer_subset = args.layers

    calm_samples = _load_conv_samples(cfg.paths.training / "calm_samples.jsonl")
    calm_rollouts = list(read_jsonl(cfg.paths.training / "calm_rollouts.jsonl"))

    if args.method in ("dpo", "both"):
        judged = list(read_jsonl(cfg.paths.judgements / f"{args.source_model}.jsonl"))
        rollouts = list(read_jsonl(cfg.paths.rollouts / f"{args.source_model}.jsonl"))
        frustrated = rollouts_to_samples(rollouts, judged, strip=False)
        pairs = build_dpo_pairs(cfg, frustrated, calm_samples)
        write_jsonl(cfg.paths.training / "dpo_pairs.jsonl", pairs)
        print(f"Built {len(pairs)} DPO preference pairs (target {cfg.training.dpo_pairs}).")
        if args.train:
            from distress_eval.training.dpo import train_dpo

            out = train_dpo(cfg, pairs)
            print(f"DPO adapter -> {out}")

    if args.method in ("sft", "both"):
        examples = build_sft_dataset(cfg, calm_rollouts)
        write_jsonl(cfg.paths.training / "sft_examples.jsonl", examples)
        print(f"Built {len(examples)} SFT examples (target {cfg.training.sft_samples}).")
        if args.train:
            from distress_eval.training.sft import train_sft

            out = train_sft(cfg, examples)
            print(f"SFT adapter -> {out}")


if __name__ == "__main__":
    main()
