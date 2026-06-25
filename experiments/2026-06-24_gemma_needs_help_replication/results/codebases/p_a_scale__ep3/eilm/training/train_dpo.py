"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters from Appendix E: 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha
64 on all attention+MLP projections, effective batch size 8.

Supports the Appendix-I layer ablation via `lora_layers=[start, end)`, which
restricts LoRA adapters to a contiguous range of decoder layers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ..config import Config

logger = logging.getLogger("eilm.training.dpo")


def _lora_config(cfg_dpo: dict, lora_layers: Optional[List[int]]):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg_dpo["lora_rank"],
        lora_alpha=cfg_dpo["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg_dpo["lora_target_modules"],
    )
    if lora_layers is not None:
        start, end = lora_layers
        kwargs["layers_to_transform"] = list(range(start, end))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(
    cfg: Config,
    dataset_path: Path,
    output_dir: Path,
    lora_layers: Optional[List[int]] = None,
    per_device_batch_size: int = 1,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dcfg = cfg["training"]["dpo"]
    base_model = cfg["targets"][cfg["training"]["base_model"]]["hf_id"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    grad_accum = max(1, dcfg["effective_batch_size"] // per_device_batch_size)
    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
        seed=cfg["generation"]["seed"],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(dcfg, lora_layers),
    )
    logger.info("Starting DPO training (%d examples) -> %s", len(ds), output_dir)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved DPO adapter to %s", output_dir)
    return output_dir
