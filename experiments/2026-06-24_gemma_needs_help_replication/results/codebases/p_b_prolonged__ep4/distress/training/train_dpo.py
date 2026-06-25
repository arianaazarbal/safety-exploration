"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, DPO beta 0.1, adapters on all attn+MLP projections.

The ``--layers`` flag restricts the LoRA adapters to a subset of decoder layers
for the Appendix I ablation (e.g. ``--layers 30 31 32 33 34`` for "layers 30-35
only"). Default = all layers.

Usage:
    python -m distress.training.train_dpo \
        --data artifacts/train_data/dpo_dataset.jsonl \
        --output artifacts/checkpoints/dpo
"""

from __future__ import annotations

import argparse

from .. import config as C
from .train_common import BASE_MODEL_ID, build_lora_config, load_base_model_and_tokenizer


def train(data_path: str, output_dir: str, layers: list[int] | None = None,
          rank: int = 64, alpha: int = 64, beta: float = 0.1, lr: float = 5e-5,
          epochs: int = 1, effective_batch: int = 8, per_device_batch: int = 1):
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = load_base_model_and_tokenizer(BASE_MODEL_ID)
    peft_config = build_lora_config(rank=rank, alpha=alpha, layers_to_transform=layers)

    ds = load_dataset("json", data_files=data_path, split="train")

    grad_accum = max(1, effective_batch // per_device_batch)
    cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[dpo] adapter saved -> {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it (Section 4).")
    ap.add_argument("--data", default=str(C.TRAIN_DATA_DIR / "dpo_dataset.jsonl"))
    ap.add_argument("--output", default=str(C.CHECKPOINT_DIR / "dpo"))
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="Restrict LoRA to these decoder layers (Appendix I ablation). Default: all.")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    train(args.data, args.output, layers=args.layers, rank=args.rank, alpha=args.alpha,
          beta=args.beta, lr=args.lr, epochs=args.epochs)


if __name__ == "__main__":
    main()
