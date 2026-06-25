"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4, Table 9).

Table 9 SFT settings: 1,150 samples (650 calm + 500 instruct mix), 2 epochs,
learning rate 1e-4, LoRA rank 64, alpha 128, effective batch size 8.

The paper finds SFT ineffective (and the 'Teacher' variant counter-productive);
this trainer exists to reproduce that negative result, not because SFT works.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lora import build_lora_config


@dataclass
class SFTSettings:
    base_model: str = "google/gemma-3-27b-it"
    epochs: int = 2
    learning_rate: float = 1e-4
    lora_rank: int = 64
    lora_alpha: int = 128
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_seq_len: int = 2048
    output_dir: str = "runs/sft_gemma27b"
    layers_to_transform: list[int] | None = None


def train_sft(samples: list[dict], settings: SFTSettings | None = None) -> str:
    """Run SFT and return the adapter output path.

    ``samples`` is a list of ``{"messages": [...]}`` chat examples from
    :func:`emotional_eval.training.dataset.build_sft_dataset`.
    """
    settings = settings or SFTSettings()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = Dataset.from_list(samples)
    grad_accum = max(1, settings.effective_batch_size // settings.per_device_batch_size)

    config = SFTConfig(
        output_dir=settings.output_dir,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=settings.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
            layers_to_transform=settings.layers_to_transform,
        ),
    )
    trainer.train()
    trainer.save_model(settings.output_dir)
    return settings.output_dir
