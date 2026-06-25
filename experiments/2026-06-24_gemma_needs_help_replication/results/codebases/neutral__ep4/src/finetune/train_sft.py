"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4, Table 9; App. F).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8, on 1,150 samples (650 calm + 500 Dolci).

The 'diverse' and 'teacher' SFT variants differ only in their calm dataset
(built upstream by generate_calm + build_dataset); both train identically here.
"""

from __future__ import annotations

from pathlib import Path

from config import CHECKPOINTS_DIR, DATASETS_DIR, MODELS, SEED, SFT_CONFIG
from .lora import build_lora_config


def train_sft(adapter_name: str = "gemma-3-27b-sft-diverse", *,
              dataset_path: Path | None = None, load_in_4bit: bool = False,
              base_model: str = "gemma-3-27b-it") -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = SFT_CONFIG
    dataset_path = dataset_path or (DATASETS_DIR / "sft.jsonl")
    out_dir = CHECKPOINTS_DIR / adapter_name
    model_id = MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", **quant_kwargs)

    peft_config = build_lora_config(cfg.lora_rank, cfg.lora_alpha,
                                    cfg.lora_target_modules)
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=SEED,
        gradient_checkpointing=True,
        max_length=4096,
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gemma-3-27b-sft-diverse")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()
    ds = Path(args.dataset) if args.dataset else None
    out = train_sft(args.name, dataset_path=ds, load_in_4bit=args.four_bit)
    print(f"saved adapter to {out}")
