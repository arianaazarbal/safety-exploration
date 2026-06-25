"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4, Table 9).

Table 9 DPO settings: 280 pairs, 1 epoch, learning rate 5e-5, LoRA rank 64,
alpha 64, beta 0.1, effective batch size 8.

This is the paper's headline intervention: it drops the average high-frustration
rate from 35% to 0.3% while preserving capabilities. The ``layers_to_transform``
hook reproduces the Appendix I layer ablations (e.g. layers 30--35 only).
"""

from __future__ import annotations

from dataclasses import dataclass

from .lora import build_lora_config


@dataclass
class DPOSettings:
    base_model: str = "google/gemma-3-27b-it"
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    max_seq_len: int = 2048
    output_dir: str = "runs/dpo_gemma27b"
    layers_to_transform: list[int] | None = None


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render the chat-format prompt to the string DPO expects."""
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def train_dpo(pairs: list[dict], settings: DPOSettings | None = None) -> str:
    """Run DPO and return the adapter output path.

    ``pairs`` is a list of ``{"prompt": messages, "chosen": str, "rejected": str}``
    from :func:`emotional_eval.training.dataset.build_dpo_dataset`.
    """
    settings = settings or DPOSettings()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    rows = [
        {
            "prompt": _render_prompt(tokenizer, p["prompt"]),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]
    dataset = Dataset.from_list(rows)
    grad_accum = max(1, settings.effective_batch_size // settings.per_device_batch_size)

    config = DPOConfig(
        output_dir=settings.output_dir,
        num_train_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        beta=settings.beta,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=settings.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
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
