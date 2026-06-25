"""DPO LoRA finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, DPO beta 0.1, adapters on all layers.

`layers` restricts adapters to a subset of decoder layers for the Appendix I
layer ablation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .lora import make_lora_config

BASE_MODEL = "google/gemma-3-27b-it"


@dataclass
class DPOHyperParams:
    learning_rate: float = 5e-5
    epochs: int = 1
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    beta: float = 0.1
    max_length: int = 4096
    max_prompt_length: int = 3072


def _load_pairs(dpo_path: str, tokenizer) -> "list[dict]":
    rows = []
    with open(dpo_path, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True
            )
            rows.append({
                "prompt": prompt,
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            })
    return rows


def train(
    dpo_path: str,
    output_dir: str,
    hp: DPOHyperParams | None = None,
    layers: list[int] | None = None,
    base_model: str = BASE_MODEL,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = hp or DPOHyperParams()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model_kwargs: dict = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    peft_config = make_lora_config(hp.lora_rank, hp.lora_alpha, layers=layers)

    dataset = Dataset.from_list(_load_pairs(dpo_path, tokenizer))
    grad_accum = max(1, hp.effective_batch_size // hp.per_device_batch_size)

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=hp.beta,
        max_length=hp.max_length,
        max_prompt_length=hp.max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
