"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. The paper finds SFT ineffective (the 'diverse' variant) or
counterproductive (the 'teacher' variant); both are trainable here for comparison.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, model_entry
from .dpo_train import _lora_config


def train_sft(cfg: Config, variant: str = "diverse",
              base_model: str = "gemma-3-27b-it") -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    s = cfg.training.sft
    model_id = model_entry(cfg, base_model)["model_id"]
    out_dir = cfg.get_path("training") / f"sft_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, cfg.local.dtype), device_map=cfg.local.device,
    )

    dataset = load_dataset(
        "json", data_files=str(cfg.get_path("datasets") / f"sft_{variant}.jsonl"),
        split="train")

    per_device = 1
    grad_accum = max(1, s.effective_batch_size // per_device)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=s.epochs,
        learning_rate=s.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=cfg.local.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
        max_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg, s.lora_rank, s.lora_alpha, None),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"SFT ({variant}) adapter saved to {out_dir}")
    return out_dir
