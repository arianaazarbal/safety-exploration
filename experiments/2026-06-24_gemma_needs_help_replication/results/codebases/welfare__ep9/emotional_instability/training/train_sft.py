"""LoRA SFT finetuning of Gemma-3-27B-it (paper Section 4.1, Appendix E).

Hyperparameters (Table 9): 1150 samples (650 calm + 500 instruct-mix), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections, effective
batch size 8.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..utils import read_jsonl


def train_sft(*, variant: str = "diverse",
              base_model: str = "google/gemma-3-27b-it",
              dataset_path: Path | None = None,
              output_dir: Path | None = None,
              cfg=config.SFT):
    """Run LoRA SFT on chat-formatted calm+instruct data."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    dataset_path = dataset_path or (config.DATA_DIR / f"sft_{variant}.jsonl")
    output_dir = output_dir or (config.CHECKPOINT_DIR / f"gemma-3-27b-sft-{variant}")
    output_dir = Path(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    rows = list(read_jsonl(dataset_path))  # each: {"messages": [...]}
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        bias="none",
    )

    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    sft_args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        # TRL applies the chat template to the "messages" field automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
