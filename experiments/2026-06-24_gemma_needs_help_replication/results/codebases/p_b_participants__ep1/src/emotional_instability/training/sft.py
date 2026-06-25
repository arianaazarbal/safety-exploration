"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1).

Paper config: 2 epochs, lr 1e-4, LoRA rank-64 on all layers, calm data mixed with
Dolci-Instruct-SFT. The paper finds SFT is *ineffective* (it does not reduce distress,
and the 'Teacher' variant marginally increases it) — we implement it faithfully anyway
as the negative control that motivates DPO.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import ExperimentConfig, ModelRegistry
from ..utils import ensure_dir

log = logging.getLogger("emotional_instability.training.sft")


def train_sft(
    sft_dataset_path: str | Path,
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    target_model: str = "gemma-3-27b-it",
    out_dir: str | Path = "artifacts/section4/sft_adapter",
) -> Path:
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from trl import SFTConfig, SFTTrainer

    sec = cfg.section("section4")["sft"]
    spec = registry.get(target_model)
    out_dir = ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(spec.id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = load_dataset("json", data_files=str(sft_dataset_path), split="train")

    peft_config = LoraConfig(
        r=int(sec["lora_rank"]),
        lora_alpha=int(sec["lora_rank"]) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # "all layers": target every linear projection in attention + MLP
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(sec["epochs"]),
        learning_rate=float(sec["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        bf16=True,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("SFT adapter saved -> %s", out_dir)
    return Path(out_dir)
