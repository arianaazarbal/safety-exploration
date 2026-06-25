"""SFT finetuning of Gemma-3-27b-it (Section 4 / Appendix E/F).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct mix), 2
epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all projection layers, effective
batch size 8. Two dataset variants ("diverse" and "teacher") differ only in how
the calm data was generated (see :mod:`generate_calm_data`).

Uses TRL's ``SFTTrainer`` on conversational-format JSONL.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config, ModelRegistry, env
from .lora_layers import lora_config

log = logging.getLogger(__name__)


def train_sft(
    dataset_jsonl: str | Path,
    cfg: Config | None = None,
    registry: ModelRegistry | None = None,
    output_dir: str | None = None,
    variant: str = "diverse",
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or Config.load("training")
    registry = registry or ModelRegistry()
    scfg = cfg.get("sft", {})
    base_name = scfg.get("base_model", "gemma-3-27b-it")
    spec = registry.target(base_name)
    vcfg = scfg.get("variants", {}).get(variant, {})
    output_dir = output_dir or vcfg.get("output_dir", f"outputs/sft_{variant}/gemma-3-27b-it")

    token = env("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )

    lcfg = scfg.get("lora", {})
    peft_cfg = lora_config(
        r=int(lcfg.get("r", 64)),
        alpha=int(lcfg.get("alpha", 128)),
        target_modules=lcfg.get("target_modules"),
        layers=lcfg.get("layers", "all"),
    )

    dataset = load_dataset("json", data_files=str(dataset_jsonl), split="train")

    eff_bs = int(scfg.get("effective_batch_size", 8))
    per_device = 1
    grad_accum = max(1, eff_bs // per_device)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=int(scfg.get("epochs", 2)),
        learning_rate=float(scfg.get("learning_rate", 1e-4)),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved SFT (%s) adapter to %s", variant, output_dir)
    return output_dir
