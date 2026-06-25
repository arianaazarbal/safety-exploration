"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. Trains on 650 calm responses + 500 Dolci-Instruct-SFT
samples. Two variants (Appendix F): 'diverse' and 'teacher'.
"""

from __future__ import annotations

from ..config import (CHECKPOINTS_DIR, GEMMA_27B_IT, LORA_TARGET_MODULES, SFT,
                      TRAIN_GRAD_ACCUM, TRAIN_MICRO_BATCH)
from .build_sft_dataset import SFT_DATASET


def train_sft(variant: str = "diverse", output_key: str | None = None):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    output_key = output_key or f"gemma-3-27b-sft-{variant}"
    data_path = SFT_DATASET.with_name(SFT_DATASET.name.format(variant=variant))

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.hf_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    peft_config = LoraConfig(
        r=SFT.lora_rank, lora_alpha=SFT.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    out_dir = CHECKPOINTS_DIR / output_key
    args = TRLSFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=SFT.epochs,
        learning_rate=SFT.learning_rate,
        per_device_train_batch_size=TRAIN_MICRO_BATCH,
        gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        # SFT on the assistant turns of conversational ("messages") data.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[sft-{variant}] saved LoRA adapter -> {out_dir}")
    return out_dir
