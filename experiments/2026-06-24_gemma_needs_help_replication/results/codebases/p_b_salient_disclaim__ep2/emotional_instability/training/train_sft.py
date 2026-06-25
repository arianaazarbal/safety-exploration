"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4, Appendix E Table 9).

Hyperparameters (Table 9):
    dataset size      1,150 samples (650 calm + 500 Dolci)
    epochs            2
    learning rate     1e-4
    LoRA rank         64
    LoRA alpha        128
    effective batch   8
    LoRA targets      q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .train_dpo import LORA_TARGET_MODULES


@dataclass
class SFTHyperParams:
    learning_rate: float = 1e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8     # effective batch size 8
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    max_length: int = 4096
    bf16: bool = True


def _load_sft_examples(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def train_sft(
    base_model_id: str,
    sft_dataset_path: Path,
    output_dir: Path,
    *,
    hp: Optional[SFTHyperParams] = None,
):
    """Run 2 epochs of LoRA SFT and save the adapter to `output_dir`."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = hp or SFTHyperParams()
    examples = _load_sft_examples(sft_dataset_path)
    ds = Dataset.from_list(examples)  # each row: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if hp.bf16 else torch.float32,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=hp.learning_rate,
        num_train_epochs=hp.num_train_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        max_length=hp.max_length,
        bf16=hp.bf16,
        logging_steps=10,
        save_strategy="epoch",
        # TRL applies the chat template to the "messages" column automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
