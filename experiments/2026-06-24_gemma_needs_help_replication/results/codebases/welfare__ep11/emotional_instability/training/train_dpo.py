"""DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E, Table 9).

1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attention+MLP projections,
beta 0.1, effective batch size 8. Trains on the 280 preference pairs and saves
the LoRA adapter to ``outputs/checkpoints/gemma-3-27b-dpo``.
"""

from __future__ import annotations

from ..config import (CHECKPOINTS_DIR, DPO, GEMMA_27B_IT, LORA_TARGET_MODULES,
                      TRAIN_GRAD_ACCUM, TRAIN_MICRO_BATCH)
from .build_dpo_dataset import DPO_DATASET


def train_dpo(output_key: str = "gemma-3-27b-dpo"):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = LoraConfig(
        r=DPO.lora_rank, lora_alpha=DPO.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    dataset = load_dataset("json", data_files=str(DPO_DATASET), split="train")

    out_dir = CHECKPOINTS_DIR / output_key
    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=DPO.epochs,
        learning_rate=DPO.learning_rate,
        beta=DPO.beta,
        per_device_train_batch_size=TRAIN_MICRO_BATCH,
        gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[dpo] saved LoRA adapter -> {out_dir}")
    return out_dir
