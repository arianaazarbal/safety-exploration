"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

Hyperparameters (Table 9): 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 /
alpha 128 on all projections, effective batch size 8. Two variants: 'diverse'
and 'teacher' (Appendix F), selected by the dataset built in build_dataset.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from .train_dpo import LORA_TARGET_MODULES


def train_sft(
    mode: str = "diverse",
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    base_model: str = "google/gemma-3-27b-it",
    epochs: int = 2,
    lr: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset_path = Path(dataset_path or (config.DATA_DIR / f"sft_{mode}.json"))
    output_dir = Path(output_dir or (config.CHECKPOINTS_DIR / f"sft_{mode}"))
    examples = json.loads(Path(dataset_path).read_text())
    ds = Dataset.from_list(examples)  # conversational: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant)

    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES)

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=config.SEED,
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_sft:{mode}] saved adapter -> {output_dir}")
    return output_dir
