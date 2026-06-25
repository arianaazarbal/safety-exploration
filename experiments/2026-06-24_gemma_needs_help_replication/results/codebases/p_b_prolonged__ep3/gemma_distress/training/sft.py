"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E/F, Table 9).

650 calm responses + 500 Dolci-Instruct-SFT samples, 2 epochs, lr 1e-4, LoRA
rank 64 / alpha 128. Two variants: 'diverse' (main text) and 'teacher'
(Appendix F). Uses ``trl.SFTTrainer`` + PEFT.

The paper reports SFT is ineffective (and the teacher variant slightly
*increases* frustration); this trainer exists to reproduce that negative result
(Figure 5, Appendix F).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from ..models import registry
from .lora import build_lora_config


def train_sft(
    variant: str = config.SFT_DIVERSE_VARIANT,
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    grad_accum: int = 8,
    dtype: str = "bfloat16",
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    dataset_path = dataset_path or (config.DATA_DIR / "train" / f"sft_{variant}.jsonl")
    output_dir = output_dir or (config.CHECKPOINT_DIR / f"sft_{variant}")

    base_id = registry.REGISTRY[registry.DPO_TARGET].identifier
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=getattr(torch, dtype))

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    peft_config = build_lora_config(config.SFT.lora)

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    # SFTTrainer applies the chat template to the "messages" field automatically.
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
