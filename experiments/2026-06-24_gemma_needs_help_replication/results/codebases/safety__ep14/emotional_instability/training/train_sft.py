"""SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8.

The paper reports SFT is ineffective (and the 'teacher' variant worsens things);
we implement it faithfully as the documented negative result / baseline.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RUNS_DIR
from .lora import make_lora_config


def train_sft(
    base_model_id: str = "google/gemma-3-27b-it",
    sft_data_path: str | Path = RUNS_DIR / "training" / "sft_data.jsonl",
    output_dir: str | Path = RUNS_DIR / "sft",
    *,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    target_modules=None,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    max_length: int = 4096,
    bf16: bool = True,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto",
    )
    peft_config = make_lora_config(lora_rank, lora_alpha, target_modules)

    dataset = load_dataset("json", data_files=str(sft_data_path), split="train")
    grad_accum = max(1, effective_batch_size // per_device_batch_size)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL applies the chat template to the "messages" field automatically.
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[sft] saved adapter -> {adapter_dir}")
    return adapter_dir
