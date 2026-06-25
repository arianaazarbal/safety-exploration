"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Table 9).

Hyperparameters (Appendix E): LoRA rank 64, alpha 128, adapters on all attention
and MLP projection layers; 2 epochs; lr 1e-4; effective batch 8.

Trains on the chat-format SFT dataset (calm + Dolci mix). The 'diverse' vs
'teacher' variants differ only in the calm-data source (Appendix F); pass the
corresponding dataset built by build_sft_dataset.py / gen_calm_data --style.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config
from ..config import SFT, FINETUNE_BASE, env, get_subject


def train(data_path: Path, out_dir: Path, *, epochs: int = SFT.epochs,
          lr: float = SFT.learning_rate):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = get_subject(FINETUNE_BASE)
    token = env("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)

    peft_cfg = LoraConfig(
        r=SFT.lora.rank, lora_alpha=SFT.lora_alpha,
        lora_dropout=SFT.lora.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=list(SFT.lora.target_modules),
    )

    ds = load_dataset("json", data_files=str(data_path), split="train")
    # Dataset rows have a "messages" field; SFTTrainer applies the chat template.

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=SFT.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        gradient_checkpointing=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved SFT LoRA adapter -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it.")
    p.add_argument("--data", required=True, help="sft_data.jsonl")
    p.add_argument("--epochs", type=int, default=SFT.epochs)
    p.add_argument("--lr", type=float, default=SFT.learning_rate)
    p.add_argument("--out", default=str(config.CKPT_DIR / "gemma27b-sft"))
    args = p.parse_args(argv)
    train(Path(args.data), Path(args.out), epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
