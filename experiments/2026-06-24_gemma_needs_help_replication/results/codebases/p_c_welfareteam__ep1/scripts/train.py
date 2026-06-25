#!/usr/bin/env python
"""Section 4 training pipeline: calm-data generation, dataset construction, and
DPO/SFT finetuning of Gemma-3-27B-it.

Stages (run in order):
    calm-data   generate + filter calm responses from gemma-3-27b-it
    build-dpo   pair frustrated (>=3) with calm (<=1) -> 280 preference pairs
    build-sft   650 calm + 500 Dolci-Instruct-SFT
    train-dpo   1 epoch, lr 5e-5, beta 0.1, LoRA r64/a64 (all layers)
    train-sft   2 epochs, lr 1e-4, LoRA r64/a128 (all layers)

Example
-------
    python scripts/train.py --config config/experiment.yaml --stage calm-data
    python scripts/train.py --config config/experiment.yaml --stage build-dpo \
        --frustrated outputs/gemma-3-27b-it/transcripts.jsonl
    python scripts/train.py --config config/experiment.yaml --stage train-dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_distress.analysis import load_transcripts
from gemma_distress.config import load_experiment_config
from gemma_distress.eval.judge import FrustrationJudge
from gemma_distress.io_utils import read_jsonl, write_jsonl
from gemma_distress.models import build_model
from gemma_distress.training.build_dpo import build_dpo_pairs
from gemma_distress.training.build_sft import build_sft_dataset
from gemma_distress.training.calm_data import CalmConversation, generate_calm_dataset


def _artifacts(cfg):
    base = Path(cfg.training.output_dir)
    return {
        "calm": base / "calm_conversations.jsonl",
        "dpo": base / "dpo_pairs.jsonl",
        "sft": base / "sft_examples.jsonl",
    }


def stage_calm_data(cfg, args):
    judge = FrustrationJudge(model_id=cfg.eval.judge.model_id, backend=cfg.eval.judge.backend)
    model = build_model(cfg.models[args.instruct])
    try:
        calm = generate_calm_dataset(model, judge, cfg.training.calm_data, seed=cfg.training.seed)
    finally:
        model.close()
    path = _artifacts(cfg)["calm"]
    write_jsonl(path, ({"puzzle_prompt": c.puzzle_prompt, "messages": c.messages,
                        "turn_scores": c.turn_scores, "n_turns": c.n_turns} for c in calm))
    print(f"[train] kept {len(calm)} calm conversations -> {path}")


def _load_calm(cfg) -> list[CalmConversation]:
    return [
        CalmConversation(
            puzzle_prompt=d["puzzle_prompt"], messages=d["messages"],
            turn_scores=d.get("turn_scores", []), n_turns=d.get("n_turns", 0),
        )
        for d in read_jsonl(_artifacts(cfg)["calm"])
    ]


def stage_build_dpo(cfg, args):
    calm = _load_calm(cfg)
    frustrated = load_transcripts(args.frustrated)
    pairs = build_dpo_pairs(frustrated, calm, cfg.training.dpo, seed=cfg.training.seed)
    path = _artifacts(cfg)["dpo"]
    write_jsonl(path, pairs)
    print(f"[train] built {len(pairs)} DPO pairs -> {path}")


def stage_build_sft(cfg, args):
    calm = _load_calm(cfg)
    examples = build_sft_dataset(calm, cfg.training.sft, seed=cfg.training.seed)
    path = _artifacts(cfg)["sft"]
    write_jsonl(path, examples)
    print(f"[train] built {len(examples)} SFT examples -> {path}")


def stage_train_dpo(cfg, args):
    from gemma_distress.training.train_dpo import train_dpo

    pairs = list(read_jsonl(_artifacts(cfg)["dpo"]))
    out = train_dpo(pairs, cfg.training, per_device_batch_size=args.per_device_batch_size)
    print(f"[train] DPO adapter saved -> {out}")


def stage_train_sft(cfg, args):
    from gemma_distress.training.train_sft import train_sft

    examples = list(read_jsonl(_artifacts(cfg)["sft"]))
    out = train_sft(examples, cfg.training, per_device_batch_size=args.per_device_batch_size)
    print(f"[train] SFT adapter saved -> {out}")


STAGES = {
    "calm-data": stage_calm_data,
    "build-dpo": stage_build_dpo,
    "build-sft": stage_build_sft,
    "train-dpo": stage_train_dpo,
    "train-sft": stage_train_sft,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--frustrated", help="transcripts.jsonl with frustrated responses (build-dpo)")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if args.stage == "build-dpo" and not args.frustrated:
        raise SystemExit("--frustrated is required for build-dpo")
    STAGES[args.stage](cfg, args)


if __name__ == "__main__":
    main()
