"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 /
alpha 128, effective batch size 8, adapters on all attention + MLP projections.

The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
this is included so that failure can be reproduced. Train on completions only
(mask the prompt) via trl.SFTTrainer's chat formatting.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import DATA_DIR, RESULTS_DIR
from .train_dpo import LORA_TARGET_MODULES


def train_sft(
    base_model: str = "google/gemma-3-27b-it",
    sft_path: str | Path = None,
    *,
    output_dir: str | Path = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    load_in_4bit: bool = True,
    hf_token: str | None = None,
    max_length: int = 4096,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from ..config import API

    sft_path = Path(sft_path or (DATA_DIR / "sft_dataset.jsonl"))
    output_dir = Path(output_dir or (RESULTS_DIR / "checkpoints" / "sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = hf_token or API.hf_token

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, token=hf_token)

    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES)

    rows = []
    with open(sft_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        # Train on completions only so calm *style* is learned, not the prompts.
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
