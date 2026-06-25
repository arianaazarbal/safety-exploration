"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1).

Paper config: 1 epoch, lr 5e-5, LoRA rank-64 on all layers, 280 (calm vs frustrated)
pairs. This is the headline mitigation — it drops avg % high-frustration from 35% to
0.3%.

Layer ablation hook: the paper finds the intervention must act on EARLY layers (adapters
from layer 40+ do not reduce distress; layers 30-35 alone are nearly as effective). The
`lora_layers` argument restricts which decoder layers get adapters so that ablation
(Section 4.2 "internal vs expressed emotions") can be reproduced.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from ..config import ExperimentConfig, ModelRegistry
from ..utils import ensure_dir

log = logging.getLogger("emotional_instability.training.dpo")


def train_dpo(
    dpo_dataset_path: str | Path,
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    target_model: str = "gemma-3-27b-it",
    out_dir: str | Path = "artifacts/section4/dpo_adapter",
    lora_layers: Optional[Sequence[int]] = None,
) -> Path:
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from trl import DPOConfig, DPOTrainer

    sec = cfg.section("section4")["dpo"]
    spec = registry.get(target_model)
    out_dir = ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(spec.id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = load_dataset("json", data_files=str(dpo_dataset_path), split="train")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    layers_to_transform = list(lora_layers) if lora_layers is not None else None

    peft_config = LoraConfig(
        r=int(sec["lora_rank"]),
        lora_alpha=int(sec["lora_rank"]) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,  # None => all layers (paper default)
    )

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(sec["epochs"]),
        learning_rate=float(sec["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        bf16=True,
        beta=0.1,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("DPO adapter saved -> %s (layers=%s)", out_dir, layers_to_transform or "all")
    return Path(out_dir)
