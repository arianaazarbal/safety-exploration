"""SFT fine-tuning baseline for Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

650 calm responses mixed with 500 Dolci-Instruct-SFT samples, 2 epochs, lr 1e-4,
LoRA rank-64 / alpha-128 on all attention + MLP projections, effective batch 8.

This is the baseline the paper finds *ineffective* (and, for the 'teacher' variant,
counterproductive); we replicate it to reproduce that negative result.
"""

from __future__ import annotations

from pathlib import Path

from ..config import MODELS, SFT
from .lora import make_lora_config


def train_sft(
    sft_path: Path,
    output_dir: Path,
    *,
    base_model: str = "gemma-3-27b-it",
    epochs: int = SFT.epochs,
    learning_rate: float = SFT.learning_rate,
) -> Path:
    import torch
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = make_lora_config(SFT.lora_rank, SFT.lora_alpha, SFT.target_modules)
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=str(sft_path), split="train")

    per_device_bs = 1
    grad_accum = max(1, SFT.effective_batch_size // per_device_bs)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        # TRL renders the {"messages": [...]} rows via the model's chat template.
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return output_dir
