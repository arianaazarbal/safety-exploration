"""DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all proj layers,
effective batch size 8. Trains on the 280 preference pairs from build_pairs.py.

`--layers 30-35` reproduces the Appendix I layer-subset ablation.
"""
from __future__ import annotations

import argparse

import config
from finetune.common import (load_base_model_and_tokenizer, load_jsonl_dataset,
                             make_lora_config, parse_layers)


def train(pairs_path: str, output_dir: str, layers: list[int] | None = None):
    from trl import DPOConfig, DPOTrainer

    model, tok, _ = load_base_model_and_tokenizer()
    dataset = load_jsonl_dataset(pairs_path)
    peft_config = make_lora_config(rank=64, alpha=64, layers=layers)

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch size 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,
        max_prompt_length=1536,
        gradient_checkpointing=True,
        report_to=[],
        seed=config.GLOBAL_SEED,
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_dpo] saved adapter -> {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(config.FINETUNE_DIR / "dpo_pairs.jsonl"))
    ap.add_argument("--out", default=str(config.FINETUNE_DIR / "dpo_adapter"))
    ap.add_argument("--layers", default=None,
                    help="restrict LoRA to layer subset, e.g. '30-35' (Appendix I)")
    args = ap.parse_args()
    train(args.pairs, args.out, parse_layers(args.layers))


if __name__ == "__main__":
    main()
