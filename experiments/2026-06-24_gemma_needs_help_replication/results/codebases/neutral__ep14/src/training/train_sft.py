"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8, adapters on all
attention + MLP projections.

The paper tests two calm-data variants ('diverse' and 'teacher', Appendix F).
Both train with identical hyperparameters; only the calm-data generation differs
(see generate_calm_data.py / training_prompts.TEACHER_SYSTEM_PROMPT), so the same
trainer covers both.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import CHECKPOINTS_DIR, FINETUNE_BASE, HF_TOKEN_ENV, get_env
from src.training.train_dpo import LORA_TARGET_MODULES


def train_sft(
    sft_jsonl: Path,
    *,
    output_dir: Path | None = None,
    base_spec=FINETUNE_BASE,
    epochs: int = 2,
    lr: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
    variant: str = "diverse",
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or (CHECKPOINTS_DIR / f"sft_{variant}_gemma27b")
    token = get_env(HF_TOKEN_ENV)
    tokenizer = AutoTokenizer.from_pretrained(base_spec.model_id, token=token)

    rows = [json.loads(l) for l in open(sft_jsonl) if l.strip()]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.model_id, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        token=token, **quant,
    )

    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM", bias="none",
    )

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_length=4096,
        # train only on assistant turns where supported by the template
        assistant_only_loss=False,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
