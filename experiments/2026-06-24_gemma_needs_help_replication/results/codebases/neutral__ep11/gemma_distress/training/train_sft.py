"""LoRA SFT fine-tuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyper-parameters: 1,150 samples (650 calm + 500 instruct-mix), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all projections, effective batch size 8.

This is included for completeness / the negative result: the paper finds SFT on
calm data fails to reduce frustration (and the 'teacher' variant increases it).
"""

from __future__ import annotations

from pathlib import Path

from ..config import (CHECKPOINTS_DIR, LORA_TARGET_MODULES, SFT_CFG,
                      GEMMA_27B_IT, ModelSpec)


def train_sft(
    dataset_path: Path,
    base_spec: ModelSpec = GEMMA_27B_IT,
    output_name: str = "sft",
    cfg=SFT_CFG,
) -> Path:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = CHECKPOINTS_DIR / f"{output_name}_{base_spec.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        # dataset has a "messages" column -> SFTTrainer applies the chat template
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
