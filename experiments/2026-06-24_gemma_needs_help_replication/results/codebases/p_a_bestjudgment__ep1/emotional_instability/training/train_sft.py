"""SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4,
LoRA rank 64 / alpha 128, effective batch size 8. Adapters on all layers.

The paper reports SFT is ineffective (and the 'teacher' variant slightly
increases frustration); this trainer exists to reproduce that negative result.
Pass `teacher=True` to build from teacher-system-prompt calm data instead (the
data builder controls which calm set is used; here we just train on whatever
sft_dataset.jsonl contains).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .build_datasets import SFT_PATH
from .lora_config import make_lora_config

BASE_MODEL = "google/gemma-3-27b-it"


def _load_sft_dataset(path: Path):
    import json
    from datasets import Dataset
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return Dataset.from_list(rows)   # each row: {"messages": [...]}


def train(output_dir: Optional[str] = None,
          lr: float = 1e-4, epochs: int = 2,
          rank: int = 64, alpha: int = 128,
          per_device_batch: int = 1, grad_accum: int = 8) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or str(config.ADAPTER_DIR / "sft_gemma")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    dataset = _load_sft_dataset(SFT_PATH)
    peft_config = make_lora_config(rank=rank, alpha=alpha)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # TRL applies the chat template to the "messages" field automatically.
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return Path(output_dir)
