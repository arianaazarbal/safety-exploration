"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4, Table 9).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci instruct),
2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8.

The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
this is replicated for completeness as a negative control.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import SFT, ADAPTER_DIR, GEMMA_27B_IT, DATA_DIR


def _lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )


def train_sft(
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    cfg=SFT,
    base_model: str = GEMMA_27B_IT.model_id,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset_path = Path(dataset_path or DATA_DIR / "sft_dataset.jsonl")
    output_dir = Path(output_dir or ADAPTER_DIR / "sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.batch_size,
        max_length=cfg.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        # Train only on the assistant turns (mask the user/prompt tokens).
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
