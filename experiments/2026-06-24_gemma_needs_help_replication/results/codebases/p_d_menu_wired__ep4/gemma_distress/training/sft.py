"""SFT of Gemma-3-27B-it on calm data with LoRA (§4.1).

Config from the paper: train on 650 calm responses mixed with 500 Dolci-Instruct
samples, 2 epochs, learning rate 1e-4, LoRA rank-64 adapters on all layers.

Uses TRL's ``SFTTrainer`` + PEFT ``LoraConfig``. Heavy imports are local so the
package imports without the training stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pairs import SFTExample


@dataclass
class SFTHyperParams:
    learning_rate: float = 1e-4
    num_train_epochs: int = 2
    lora_rank: int = 64
    lora_alpha: int = 128
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8


def lora_config(hp: SFTHyperParams):
    """Rank-64 LoRA on all linear layers ("all layers" in the paper)."""
    from peft import LoraConfig

    return LoraConfig(
        r=hp.lora_rank,
        lora_alpha=hp.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def train_sft(
    base_model_id: str,
    examples: list[SFTExample],
    output_dir: str,
    hp: SFTHyperParams | None = None,
):
    """Run SFT and write the adapter to ``output_dir``. Returns the trainer."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = hp or SFTHyperParams()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = Dataset.from_list([{"messages": ex.messages} for ex in examples])

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hp.num_train_epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(hp),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer
