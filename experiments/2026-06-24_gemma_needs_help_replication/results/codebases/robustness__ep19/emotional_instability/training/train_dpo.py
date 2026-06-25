"""LoRA DPO finetuning of Gemma-3-27B-it (Table 9 / Appendix E).

Hyperparameters: 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all
attention+MLP projections, effective batch size 8. Reproducing the Appendix I
layer ablation is a one-line change via `layers_to_transform` in config.LoRAConfig.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import (ARTIFACTS_DIR, DPO_CONFIG, FINETUNE_BASE_MODEL, LoRAConfig,
                      MODELS)


def _load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_prompt(tokenizer, prompt_msgs: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)


def train_dpo(
    pairs_path: Path,
    *,
    base_model: str = FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
    cfg=DPO_CONFIG,
    lora: LoRAConfig | None = None,
    load_in_4bit: bool = False,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    lora = lora or cfg.lora
    output_dir = output_dir or (ARTIFACTS_DIR / "gemma-dpo")
    model_id = MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the trl-style preference dataset: prompt is the chat-templated
    # history; chosen/rejected are the assistant completions.
    raw = _load_pairs(pairs_path)
    rows = {"prompt": [], "chosen": [], "rejected": []}
    for p in raw:
        rows["prompt"].append(_format_prompt(tokenizer, p["prompt"]))
        rows["chosen"].append(p["chosen"])
        rows["rejected"].append(p["rejected"])
    dataset = Dataset.from_dict(rows)

    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", **quant_kwargs)

    peft_config = LoraConfig(
        r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        layers_to_transform=(list(lora.layers_to_transform)
                             if lora.layers_to_transform else None),
        task_type="CAUSAL_LM", bias="none",
    )

    # Effective batch size 8 via grad accumulation (per-device batch 1 to fit 27B).
    grad_accum = max(1, cfg.effective_batch_size)
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        max_prompt_length=1536,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir
