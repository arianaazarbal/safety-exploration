"""SFT finetuning of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct-mix), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. The paper finds SFT
ineffective (and the 'teacher' variant counterproductive); this script lets us
reproduce that negative result.

Usage:
    python -m distress.training.train_sft \
        --data artifacts/train_data/sft_dataset.jsonl \
        --output artifacts/checkpoints/sft
"""

from __future__ import annotations

import argparse

from .. import config as C
from .train_common import BASE_MODEL_ID, build_lora_config, load_base_model_and_tokenizer


def train(data_path: str, output_dir: str, rank: int = 64, alpha: int = 128,
          lr: float = 1e-4, epochs: int = 2, effective_batch: int = 8, per_device_batch: int = 1):
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = load_base_model_and_tokenizer(BASE_MODEL_ID)
    peft_config = build_lora_config(rank=rank, alpha=alpha)
    ds = load_dataset("json", data_files=data_path, split="train")

    grad_accum = max(1, effective_batch // per_device_batch)
    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        max_length=4096,
        packing=False,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[sft] adapter saved -> {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it on calm data (Section 4).")
    ap.add_argument("--data", default=str(C.TRAIN_DATA_DIR / "sft_dataset.jsonl"))
    ap.add_argument("--output", default=str(C.CHECKPOINT_DIR / "sft"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    train(args.data, args.output, rank=args.rank, alpha=args.alpha, lr=args.lr, epochs=args.epochs)


if __name__ == "__main__":
    main()
