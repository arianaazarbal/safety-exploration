"""SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E/F).

Hyperparameters (Table 9):
  dataset      1150 samples (650 calm + 500 instruct)
  epochs       2
  lr           1e-4
  LoRA rank    64,  alpha 128,  all proj layers
  eff. batch   8

The paper reports SFT is ineffective (and the 'teacher' variant worsens
frustration, Appendix F); we still implement it for the comparison in Figure 5.
The 'teacher' variant simply uses a calm pool generated with TEACHER_SYSTEM
(see generate_calm.generate_pool); training code is identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import CHECKPOINT_DIR, INTERVENTION_BASE, get_model
from .build_dataset import SFT_DATASET
from .lora import make_lora_config


def train_sft(
    *,
    dataset_path: Path = SFT_DATASET,
    output_dir: Optional[Path] = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_length: int = 4096,
    seed: int = 0,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = get_model(INTERVENTION_BASE)
    output_dir = Path(output_dir or (CHECKPOINT_DIR / "gemma27b-sft"))

    tokenizer = AutoTokenizer.from_pretrained(spec.identifier)
    model = AutoModelForCausalLM.from_pretrained(
        spec.identifier, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    peft_config = make_lora_config(rank=lora_rank, alpha=lora_alpha)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        packing=False,
        report_to=[],
        # TRL applies the chat template to the "messages" column automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
