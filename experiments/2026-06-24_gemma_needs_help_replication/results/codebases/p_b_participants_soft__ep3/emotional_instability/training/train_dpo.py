"""DPO finetuning of Gemma-3-27B-it with LoRA (Table 9 / Appendix E).

280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attn+MLP
projection modules. Supports `layers_to_transform` for the Appendix-I layer
ablation (apply adapters to a subset of layers only).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..config import DPO, LORA_TARGET_MODULES, PATHS


def _load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _render_prompt(tokenizer, prompt_field: str) -> str:
    """`prompt` is JSON-encoded chat messages -> render to a prompt string."""
    messages = json.loads(prompt_field)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def train_dpo(
    pairs_path: Optional[str] = None,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: Optional[str] = None,
    layers_to_transform: Optional[list[int]] = None,
    load_in_4bit: bool = True,
    cfg=DPO,
):
    """Run one DPO epoch and save the LoRA adapter to `output_dir`."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    pairs_path = pairs_path or os.path.join(PATHS.datasets, "dpo", "dpo_pairs.jsonl")
    suffix = "all" if not layers_to_transform else f"L{layers_to_transform[0]}-{layers_to_transform[-1]}"
    output_dir = output_dir or os.path.join(PATHS.checkpoints, f"gemma27b_dpo_{suffix}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = _load_pairs(pairs_path)
    ds = Dataset.from_list([
        {
            "prompt": _render_prompt(tokenizer, p["prompt"]),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=layers_to_transform,   # None => all layers
    )

    # effective batch size 8 (Table 9): split into per-device * grad accumulation.
    per_device = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
