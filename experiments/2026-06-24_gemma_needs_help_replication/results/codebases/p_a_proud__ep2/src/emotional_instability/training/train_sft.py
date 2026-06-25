"""LoRA SFT finetuning of Gemma-3-27B-it (§4.1, App. E, Table 9).

Hyperparameters: 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP
projections, effective batch size 8. Trains on the conversational dataset produced by
:mod:`.build_sft` (calm responses + Dolci instruct mix). The paper finds SFT ineffective
(and the 'teacher' variant counterproductive); this trains the model the paper analyses.
"""
from __future__ import annotations

from pathlib import Path

from ..config import SFTConfig
from ..utils import write_json
from .lora_layers import build_lora_config


def train_sft(
    dataset_path: str,
    output_dir: str,
    *,
    base_model: str = "google/gemma-3-27b-it",
    cfg: SFTConfig | None = None,
    per_device_batch_size: int = 1,
    bf16: bool = True,
    dtype: str = "bfloat16",
) -> dict:
    """Run SFT and save the LoRA adapter to ``output_dir``. Returns a run manifest."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    cfg = cfg or SFTConfig()
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=getattr(torch, dtype),
        device_map="auto", attn_implementation="eager",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        packing=False,
    )

    trainer = SFTTrainer(
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
        "method": "sft", "variant": cfg.variant, "base_model": base_model,
        "output_dir": output_dir, "n_examples": len(dataset), "epochs": cfg.epochs,
        "lr": cfg.learning_rate, "lora_rank": cfg.lora.rank, "lora_alpha": cfg.lora.alpha,
        "effective_batch_size": cfg.effective_batch_size,
    }
    write_json(Path(output_dir, "train_manifest.json"), manifest)
    return manifest
