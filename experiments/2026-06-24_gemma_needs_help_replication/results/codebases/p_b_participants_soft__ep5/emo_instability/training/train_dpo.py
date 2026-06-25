"""DPO finetuning of Gemma-3-27B-it (Section 4.1).

From the main text (Section 4.1): 280 preference pairs, 1 epoch, lr 5e-5, LoRA
rank-64 adapters on all layers. The remaining hyperparameters — DPO beta (0.1),
LoRA alpha (64), effective batch size (8), max lengths — are NOT given in the
provided text (they would be in Appendix E) and are set to standard, reasonable
DPO defaults here (see DESIGN.md).

Also supports the Appendix I.1 layer-subset ablation via ``layers``.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACTS_DIR, DATA_DIR, get_participant
from .lora import lora_config


def train(
    *,
    base_model: str = "gemma-3-27b-it",
    dataset_path: str | Path | None = None,
    output_name: str = "gemma-3-27b-it-dpo",
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    rank: int = 64,
    alpha: int = 64,
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    layers: list[int] | None = None,
    dtype: str = "bfloat16",
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = get_participant(base_model)
    dataset_path = str(dataset_path or (DATA_DIR / "dpo_pairs.jsonl"))
    out_dir = ARTIFACTS_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.ref)
    model = AutoModelForCausalLM.from_pretrained(
        spec.ref, torch_dtype=getattr(torch, dtype), device_map="auto"
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    # DPOTrainer expects columns: prompt, chosen, rejected (present in our jsonl).

    grad_accum = max(1, effective_batch_size // per_device_batch_size)
    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(rank=rank, alpha=alpha, layers=layers),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
