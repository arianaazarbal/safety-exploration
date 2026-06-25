"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4 / Table 9).

2 epochs, lr 1e-4, LoRA rank-64 alpha-128 on all attn+MLP projections, effective
batch size 8. Trains on 650 calm responses mixed with standard instruct data.
"""

from __future__ import annotations

from pathlib import Path

from ..config import FinetuneConfig
from ..clients.factory import model_by_name
from .datasets import SFT_PATH
from .train_dpo import _lora_config


def train(cfg: FinetuneConfig, data_path: Path = SFT_PATH) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_id = model_by_name(cfg.base_model).model_id
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Dataset rows are {"messages": [...]}; SFTTrainer applies the chat template.
    ds = load_dataset("json", data_files=str(data_path), split="train")

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[done] SFT adapter saved -> {cfg.output_dir}")
    return cfg.output_dir
