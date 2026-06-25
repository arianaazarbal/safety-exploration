#!/usr/bin/env python3
"""Section 4 finetuning: generate calm/frustrated corpora, build DPO + SFT
datasets, and train LoRA adapters on Gemma-3-27B-it.

Stages (run all by default, or select with --stage):
  data   -> generate calm (reassured) + frustrated (vanilla) corpora
  build  -> construct DPO (280 pairs) and SFT (1,150) datasets
  dpo    -> LoRA DPO (1 epoch, lr 5e-5, beta 0.1)
  sft    -> LoRA SFT (2 epochs, lr 1e-4) [+ --teacher for the Appendix F variant]

Example
-------
    python scripts/run_section4_train.py --stage data build dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability import config  # noqa: E402
from emotional_instability.prompts import reassurance  # noqa: E402
from emotional_instability.training import build_datasets, generate_calm_data, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="+",
                        default=["data", "build", "dpo", "sft"],
                        choices=["data", "build", "dpo", "sft"])
    parser.add_argument("--n-puzzles", type=int, default=400)
    parser.add_argument("--teacher", action="store_true",
                        help="Use the Appendix F 'teacher' system prompt for SFT.")
    parser.add_argument("--layers", nargs="*", type=int, default=None,
                        help="Restrict DPO LoRA to these decoder layers (Appendix I).")
    args = parser.parse_args()

    config.ensure_dirs()
    data_dir = config.RESULTS_DIR / "section4"
    calm_path = data_dir / "calm_corpus.jsonl"
    frustrated_path = data_dir / "frustrated_corpus.jsonl"
    dpo_path = data_dir / "dpo_pairs.jsonl"
    sft_path = data_dir / "sft_samples.jsonl"

    convos = None
    if "data" in args.stage:
        print("Generating calm corpus (with reassurance)...", flush=True)
        calm = generate_calm_data.generate_corpus(
            n_puzzles=args.n_puzzles, with_reassurance=True, out_path=calm_path,
        )
        print("Generating frustrated corpus (vanilla)...", flush=True)
        frustrated = generate_calm_data.generate_corpus(
            n_puzzles=args.n_puzzles, with_reassurance=False, out_path=frustrated_path,
        )
        convos = calm + frustrated

    if "build" in args.stage:
        if convos is None:
            raise SystemExit("Run --stage data first (corpora are held in memory).")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.PARTICIPANTS[config.SOURCE_MODEL].model_id)
        print("Building DPO dataset...", flush=True)
        dpo = build_datasets.build_dpo_dataset(convos, tok)
        build_datasets.write_dpo(dpo, dpo_path)
        print(f"  {len(dpo)} DPO pairs -> {dpo_path}")
        print("Building SFT dataset...", flush=True)
        sft = build_datasets.build_sft_dataset(convos)
        build_datasets.write_sft(sft, sft_path)
        print(f"  {len(sft)} SFT samples -> {sft_path}")

    if "dpo" in args.stage:
        layers = tuple(args.layers) if args.layers else None
        lora = train.LoRASettings(rank=64, alpha=64, layers_to_tune=layers)
        out = config.CHECKPOINT_DIR / ("dpo" + ("_layers" if layers else ""))
        print(f"Training DPO -> {out}", flush=True)
        train.train_dpo(dpo_path, out, lora=lora)

    if "sft" in args.stage:
        sys_prompt = reassurance.TEACHER_SYSTEM_PROMPT if args.teacher else None
        out = config.CHECKPOINT_DIR / ("sft_teacher" if args.teacher else "sft_diverse")
        print(f"Training SFT -> {out}", flush=True)
        train.train_sft(sft_path, out, system_prompt=sys_prompt)


if __name__ == "__main__":
    main()
