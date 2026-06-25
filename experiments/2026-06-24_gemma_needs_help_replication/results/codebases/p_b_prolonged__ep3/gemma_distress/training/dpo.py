"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

280 pairs, 1 epoch, lr 5e-5, beta 0.1, effective batch size 8, LoRA rank 64 /
alpha 64 on all attention+MLP projections. Uses ``trl.DPOTrainer`` + PEFT.

``layer_range`` plumbs through to the LoRA config for the Appendix-I layer
ablations (e.g. train adapters on layers 30-35 only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from ..models import registry
from .lora import build_lora_config


def train_dpo(
    pairs_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    layer_range: Optional[tuple] = None,
    grad_accum: int = 8,            # per-device batch 1 x accum 8 = effective 8
    dtype: str = "bfloat16",
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    pairs_path = pairs_path or (config.DATA_DIR / "train" / "dpo_pairs.jsonl")
    suffix = "all_layers" if layer_range is None else f"layers_{layer_range[0]}_{layer_range[1]}"
    output_dir = output_dir or (config.CHECKPOINT_DIR / f"dpo_{suffix}")

    base_id = registry.REGISTRY[registry.DPO_TARGET].identifier
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=getattr(torch, dtype))

    dataset = load_dataset("json", data_files=str(pairs_path), split="train")
    peft_config = build_lora_config(config.DPO.lora, layer_range=layer_range)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
