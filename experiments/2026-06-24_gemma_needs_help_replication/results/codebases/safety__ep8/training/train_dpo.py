"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
  dataset = 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
  effective batch size 8, DPO beta 0.1, LoRA on all attn+MLP projections.

Run:  python -m training.train_dpo --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from distress_eval.config import load_config

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base_model", default="google/gemma-3-27b-it")
    ap.add_argument("--dataset", default=None, help="path to dpo_dataset.jsonl")
    ap.add_argument("--output", default=None, help="adapter output dir")
    ap.add_argument("--layers", default="all",
                    help="'all' or comma-separated layer indices for the ablation "
                         "(e.g. '30,31,32,33,34,35'); see Section 4.2 internal-emotion result")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else \
        config.output_dir / "training" / "dpo_dataset.jsonl"
    output_dir = Path(args.output) if args.output else \
        config.output_dir / "checkpoints" / "dpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto")

    # Drop the "meta" field; TRL wants prompt/chosen/rejected only.
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ("prompt", "chosen", "rejected")])

    target_modules = LORA_TARGET_MODULES
    layers_to_transform = None
    if args.layers != "all":
        layers_to_transform = [int(x) for x in args.layers.split(",")]

    peft_config = LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,    # effective batch size 8
        beta=0.1,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved DPO LoRA adapter -> {output_dir}")


if __name__ == "__main__":
    main()
