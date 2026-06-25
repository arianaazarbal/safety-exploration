"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, App. E Table 9).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8, LoRA on all attention +
MLP projections. The paper finds SFT ineffective; we replicate it as the
comparison arm. Trained adapter saved to checkpoints/gemma-3-27b-sft.
"""

from __future__ import annotations

from pathlib import Path

from config import API, CHECKPOINTS_DIR, DATASETS_DIR
from src.finetune.train_dpo import LORA_TARGET_MODULES


def train_sft(
    *,
    base_model: str = "google/gemma-3-27b-it",
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    dataset_path = dataset_path or (DATASETS_DIR / "sft_dataset.jsonl")
    output_dir = output_dir or (CHECKPOINTS_DIR / "gemma-3-27b-sft")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=API.hf_token)

    model_kwargs: dict = {"torch_dtype": torch.bfloat16, "token": API.hf_token, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )

    # Dataset rows are {"messages": [...]}; SFTTrainer applies the chat template.
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_sft] saved adapter -> {output_dir}")
    return output_dir
