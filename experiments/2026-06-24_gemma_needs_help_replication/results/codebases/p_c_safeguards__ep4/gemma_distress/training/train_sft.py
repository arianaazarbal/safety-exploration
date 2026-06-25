"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E/F).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128. 'diverse' and 'teacher' dataset
variants (Appendix F); the teacher variant reproduces the finding that SFT can
*increase* frustration.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import get_model_spec, training_config
from .build_dataset import DATA_ROOT, load_jsonl
from .train_dpo import ADAPTER_ROOT

logger = logging.getLogger(__name__)


def _lora_config():
    from peft import LoraConfig

    cfg = training_config()
    sft = cfg["sft"]
    return LoraConfig(
        r=sft["lora_rank"],
        lora_alpha=sft["lora_alpha"],
        lora_dropout=0.0,
        target_modules=cfg["lora_target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )


def train(output_name: str | None = None, micro_batch: int = 1) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = training_config()
    sft = cfg["sft"]
    variant = sft["variant"]
    output_name = output_name or f"gemma-3-27b-it-sft-{variant}"
    spec = get_model_spec(cfg["base_model"])

    rows = load_jsonl(DATA_ROOT / f"sft_{variant}.jsonl")
    if not rows:
        raise RuntimeError(f"No SFT data for variant '{variant}'; build it first.")
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, sft["effective_batch_size"] // micro_batch)
    out_dir = ADAPTER_ROOT / output_name
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    logger.info("starting SFT (%s): %d examples", variant, len(rows))
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir
