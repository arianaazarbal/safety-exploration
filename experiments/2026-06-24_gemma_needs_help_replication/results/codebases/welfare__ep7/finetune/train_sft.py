"""SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all proj layers, effective batch
size 8. Trains on calm responses mixed with instruct data (build_pairs.py).
Per Section 4.2 this is the *ineffective* baseline; we include it to reproduce
the SFT-vs-DPO comparison (Figure 5) and the teacher-SFT failure (Appendix F).
"""
from __future__ import annotations

import argparse

import config
from finetune.common import (load_base_model_and_tokenizer, load_jsonl_dataset,
                             make_lora_config)


def train(data_path: str, output_dir: str):
    from trl import SFTConfig, SFTTrainer

    model, tok, _ = load_base_model_and_tokenizer()
    dataset = load_jsonl_dataset(data_path)        # conversational {"messages": ...}
    peft_config = make_lora_config(rank=64, alpha=128)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch size 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,
        gradient_checkpointing=True,
        report_to=[],
        seed=config.GLOBAL_SEED,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_sft] saved adapter -> {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()
    data = config.FINETUNE_DIR / f"sft_{args.variant}.jsonl"
    out = config.FINETUNE_DIR / f"sft_{args.variant}_adapter"
    train(str(data), str(out))


if __name__ == "__main__":
    main()
