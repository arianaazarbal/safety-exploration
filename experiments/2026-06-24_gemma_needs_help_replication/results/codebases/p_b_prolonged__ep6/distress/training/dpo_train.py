"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E).

Hyperparameters (Table 9):
  dataset      280 preference pairs
  epochs       1
  lr           5e-5
  LoRA rank    64,  alpha 64,  all proj layers
  eff. batch   8
  beta         0.1
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import CHECKPOINT_DIR, INTERVENTION_BASE, get_model
from .build_dataset import DPO_DATASET
from .lora import make_lora_config


def train_dpo(
    *,
    dataset_path: Path = DPO_DATASET,
    output_dir: Optional[Path] = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    lora_layers: Optional[list[int]] = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,           # effective batch size 8
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    seed: int = 0,
):
    """Run DPO and save the LoRA adapter. Returns the output directory.

    `lora_layers` restricts adapters to a contiguous layer range for the
    Appendix I ablations (e.g. list(range(30, 35)))."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = get_model(INTERVENTION_BASE)
    output_dir = Path(output_dir or (CHECKPOINT_DIR / "gemma27b-dpo"))

    tokenizer = AutoTokenizer.from_pretrained(spec.identifier)
    model = AutoModelForCausalLM.from_pretrained(
        spec.identifier, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    # TRL DPOTrainer expects columns prompt/chosen/rejected.
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ("prompt", "chosen", "rejected")])

    peft_config = make_lora_config(rank=lora_rank, alpha=lora_alpha,
                                   layers=lora_layers)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # LoRA: reference is the frozen base
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
