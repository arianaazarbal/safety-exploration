"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters from Appendix E: 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on
all attention+MLP projections, effective batch size 8. Trains on calm
conversations mixed with standard instruct data.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config

logger = logging.getLogger("eilm.training.sft")


def _lora_config(cfg_sft: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg_sft["lora_rank"],
        lora_alpha=cfg_sft["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg_sft["lora_target_modules"],
    )


def train_sft(cfg: Config, dataset_path: Path, output_dir: Path,
              per_device_batch_size: int = 1) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    scfg = cfg["training"]["sft"]
    base_model = cfg["targets"][cfg["training"]["base_model"]]["hf_id"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    grad_accum = max(1, scfg["effective_batch_size"] // per_device_batch_size)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        packing=False,
        seed=cfg["generation"]["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(scfg),
    )
    logger.info("Starting SFT training (%d examples, variant=%s) -> %s",
                len(ds), scfg["variant"], output_dir)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved SFT adapter to %s", output_dir)
    return output_dir
