"""DPO finetuning of Gemma-3-27b-it (Section 4 / Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 /
alpha 64 on all attention+MLP projection layers, effective batch size 8.

Uses TRL's ``DPOTrainer`` with a PEFT LoRA config. The dataset is the
conversational-format JSONL from :mod:`build_dpo_pairs`; TRL applies the chat
template and tokenises.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config, ModelRegistry, env
from .lora_layers import lora_config

log = logging.getLogger(__name__)


def train_dpo(
    pairs_jsonl: str | Path,
    cfg: Config | None = None,
    registry: ModelRegistry | None = None,
    output_dir: str | None = None,
    layers="all",
    num_layers: int = 50,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = cfg or Config.load("training")
    registry = registry or ModelRegistry()
    dcfg = cfg.get("dpo", {})
    base_name = dcfg.get("base_model", "gemma-3-27b-it")
    spec = registry.target(base_name)
    output_dir = output_dir or dcfg.get("output_dir", "outputs/dpo/gemma-3-27b-it")

    token = env("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )

    lcfg = dcfg.get("lora", {})
    peft_cfg = lora_config(
        r=int(lcfg.get("r", 64)),
        alpha=int(lcfg.get("alpha", 64)),
        target_modules=lcfg.get("target_modules"),
        layers=layers if layers != "all" else lcfg.get("layers", "all"),
        num_layers=num_layers,
    )

    dataset = load_dataset("json", data_files=str(pairs_jsonl), split="train")

    # Effective batch size 8 = per_device_batch_size * grad_accum (single GPU here).
    eff_bs = int(dcfg.get("effective_batch_size", 8))
    per_device = 1
    grad_accum = max(1, eff_bs // per_device)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=int(dcfg.get("epochs", 1)),
        learning_rate=float(dcfg.get("learning_rate", 5e-5)),
        beta=float(dcfg.get("beta", 0.1)),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved DPO adapter to %s", output_dir)
    return output_dir
