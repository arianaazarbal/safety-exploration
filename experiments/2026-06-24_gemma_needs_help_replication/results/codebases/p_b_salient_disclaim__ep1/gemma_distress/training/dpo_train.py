"""DPO finetuning of Gemma-3-27B-it (PAPER Section 4.1, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attn+MLP projections,
effective batch size 8. Trains on 280 preference pairs.
"""
from __future__ import annotations

import math

from ..config import experiment_config, get_target_spec


def train_dpo(
    *,
    dataset,                      # datasets.Dataset from build_dpo_dataset
    base_model: str = "gemma-3-27b-it",
    output_dir: str,
    lora_layers=None,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = experiment_config()["dpo"]
    hf_id = get_target_spec(base_model).params["hf_id"]
    layers = lora_layers if lora_layers is not None else cfg["lora_layers"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    from .lora import build_lora_config
    peft_config = build_lora_config(
        rank=cfg["lora_rank"], alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"], layers=layers,
    )

    grad_accum = max(1, cfg["effective_batch_size"] // per_device_batch_size)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # LoRA adapter is applied to a base instruct model; with PEFT, TRL uses
        # the unadapted model as the implicit reference (no separate ref model).
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = f"{output_dir}/adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
