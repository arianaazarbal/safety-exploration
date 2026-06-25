#!/usr/bin/env python
"""Section 4: calm-data generation + SFT/DPO interventions on Gemma-3-27B-it.

Subcommands:
    generate   sample calm + frustrated corpora (and the teacher calm corpus)
    dpo        build 280 preference pairs and train the DPO adapter
    sft        build the diverse SFT dataset (650 calm + 500 Dolci) and train
    teacher    train the teacher-SFT variant (Appendix F)
    all        run generate -> dpo -> sft -> teacher in sequence

    python scripts/run_section4_training.py all
"""

import argparse

import _bootstrap  # noqa: F401

import config
from emotional_instability.training.generate_calm_data import TurnRecord, generate_corpus
from emotional_instability.training.build_datasets import build_dpo_dataset, build_sft_dataset
from emotional_instability.training.train_dpo import train_dpo
from emotional_instability.training.train_sft import train_sft
from emotional_instability.utils.io import load_jsonl, write_jsonl

CORPUS_DIR = config.RESULTS_DIR / "section4"


def _save(records, path):
    write_jsonl(path, (r.to_row() for r in records))


def _load(path) -> list[TurnRecord]:
    return [TurnRecord(**row) for row in load_jsonl(path)]


def cmd_generate(args, *, teacher: bool = False) -> None:
    calm, frustrated = generate_corpus(
        teacher=teacher, seed=args.seed,
        model_kwargs={"load_in_4bit": True} if args.load_in_4bit else {},
    )
    suffix = "teacher" if teacher else "diverse"
    _save(calm, CORPUS_DIR / f"calm_{suffix}.jsonl")
    if not teacher:
        _save(frustrated, CORPUS_DIR / "frustrated.jsonl")
    print(f"calm({suffix})={len(calm)}  frustrated={len(frustrated)}")


def cmd_dpo(args) -> None:
    calm = _load(CORPUS_DIR / "calm_diverse.jsonl")
    frustrated = _load(CORPUS_DIR / "frustrated.jsonl")
    pairs = build_dpo_dataset(calm, frustrated, seed=args.seed)
    write_jsonl(CORPUS_DIR / "dpo_pairs.jsonl", pairs)
    print(f"Built {len(pairs)} DPO pairs; training adapter -> {config.DPO_ADAPTER_DIR}")
    train_dpo(pairs, output_dir=config.DPO_ADAPTER_DIR)


def cmd_sft(args, *, teacher: bool = False) -> None:
    suffix = "teacher" if teacher else "diverse"
    calm = _load(CORPUS_DIR / f"calm_{suffix}.jsonl")
    samples = build_sft_dataset(calm, seed=args.seed)
    out = config.SFT_TEACHER_ADAPTER_DIR if teacher else config.SFT_DIVERSE_ADAPTER_DIR
    print(f"Built {len(samples)} SFT samples ({suffix}); training -> {out}")
    train_sft(samples, output_dir=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "dpo", "sft", "teacher", "all"])
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    if args.command == "generate":
        cmd_generate(args)
        cmd_generate(args, teacher=True)
    elif args.command == "dpo":
        cmd_dpo(args)
    elif args.command == "sft":
        cmd_sft(args)
    elif args.command == "teacher":
        cmd_sft(args, teacher=True)
    elif args.command == "all":
        cmd_generate(args)
        cmd_generate(args, teacher=True)
        cmd_dpo(args)
        cmd_sft(args)
        cmd_sft(args, teacher=True)


if __name__ == "__main__":
    main()
