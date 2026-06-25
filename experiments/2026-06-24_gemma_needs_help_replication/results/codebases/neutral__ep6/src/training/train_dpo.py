"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64
on all attention + MLP projections, effective batch size 8, DPO beta 0.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def train_dpo(
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    base_model: str = "google/gemma-3-27b-it",
    epochs: int = 1,
    lr: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dataset_path = Path(dataset_path or (config.DATA_DIR / "dpo_pairs.json"))
    output_dir = Path(output_dir or (config.CHECKPOINTS_DIR / "dpo"))
    pairs = json.loads(Path(dataset_path).read_text())
    ds = Dataset.from_list(pairs)  # conversational: prompt/chosen/rejected

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
    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=config.SEED,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_dpo] saved adapter -> {output_dir}")
    return output_dir
