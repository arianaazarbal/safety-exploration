"""LoRA DPO finetuning of Gemma-3-27B-it (§4.1, App. E, Table 9).

Hyperparameters: 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP
projections, effective batch size 8. ``layer_range`` restricts adapters to a subset of
layers for the App. I ablations.

Uses TRL's DPOTrainer with the conversational dataset produced by :mod:`.build_dpo`.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import DPOConfig
from ..utils import write_json
from .lora_layers import build_lora_config


def train_dpo(
    dataset_path: str,
    output_dir: str,
    *,
    base_model: str = "google/gemma-3-27b-it",
    cfg: DPOConfig | None = None,
    layer_range: tuple[int, int] | None = None,
    per_device_batch_size: int = 1,
    bf16: bool = True,
    dtype: str = "bfloat16",
) -> dict:
    """Run DPO and save the LoRA adapter to ``output_dir``. Returns a run manifest."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    cfg = cfg or DPOConfig()
    if layer_range is not None:
        cfg = replace(cfg, lora=replace(cfg.lora, layer_range=layer_range))

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=getattr(torch, dtype),
        device_map="auto", attn_implementation="eager",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "method": "dpo", "base_model": base_model, "output_dir": output_dir,
        "n_examples": len(dataset), "epochs": cfg.epochs, "lr": cfg.learning_rate,
        "beta": cfg.beta, "lora_rank": cfg.lora.rank, "lora_alpha": cfg.lora.alpha,
        "layer_range": layer_range, "effective_batch_size": cfg.effective_batch_size,
    }
    write_json(Path(output_dir, "train_manifest.json"), manifest)
    return manifest
