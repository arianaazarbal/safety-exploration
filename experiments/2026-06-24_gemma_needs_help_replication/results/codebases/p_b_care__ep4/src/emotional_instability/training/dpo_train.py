"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attention+MLP projections,
beta 0.1, effective batch size 8. Optionally restrict LoRA to a subset of decoder
layers via ``target_layers`` for the Appendix I ablations.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, model_entry


def _lora_config(cfg: Config, rank: int, alpha: int, target_layers: list[int] | None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(cfg.training.lora_target_modules),
        layers_to_transform=target_layers,  # None => all layers
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_dpo(cfg: Config, base_model: str = "gemma-3-27b-it",
              output_subdir: str = "dpo", target_layers: list[int] | None = None) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    d = cfg.training.dpo
    model_id = model_entry(cfg, base_model)["model_id"]
    out_dir = cfg.get_path("training") / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, cfg.local.dtype), device_map=cfg.local.device,
    )

    dataset = load_dataset(
        "json", data_files=str(cfg.get_path("datasets") / "dpo_pairs.jsonl"), split="train")

    # Effective batch size 8 via grad accumulation (per-device batch 1 fits 27B+LoRA).
    per_device = 1
    grad_accum = max(1, d.effective_batch_size // per_device)

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=d.epochs,
        learning_rate=d.learning_rate,
        beta=d.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=cfg.local.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg, d.lora_rank, d.lora_alpha, target_layers),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"DPO adapter saved to {out_dir}")
    return out_dir
