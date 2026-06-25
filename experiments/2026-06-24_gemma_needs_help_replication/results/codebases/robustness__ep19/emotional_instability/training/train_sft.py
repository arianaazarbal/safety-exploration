"""LoRA SFT finetuning of Gemma-3-27B-it (Table 9 / Appendix E).

650 calm responses mixed with 500 Dolci-Instruct-SFT samples to mitigate
degeneration. 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all projections.
The paper reports SFT is ineffective (and the 'teacher' variant worsens
frustration, Appendix F) — this is included to reproduce that negative result.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import (ARTIFACTS_DIR, FINETUNE_BASE_MODEL, LoRAConfig, MODELS,
                      SFT_CONFIG)


def _load_calm(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_instruct_mix(n: int, dataset_name: str) -> list[dict]:
    """Load standard instruct conversations to mix in (anti-degeneration)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"[sft] could not load {dataset_name} ({e}); proceeding without mix")
        return []


def train_sft(
    calm_path: Path,
    *,
    base_model: str = FINETUNE_BASE_MODEL,
    output_dir: Path | None = None,
    cfg=SFT_CONFIG,
    lora: LoRAConfig | None = None,
    load_in_4bit: bool = False,
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    lora = lora or cfg.lora
    output_dir = output_dir or (ARTIFACTS_DIR / "gemma-sft")
    model_id = MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calm = _load_calm(calm_path)[: cfg.n_calm]
    mix = _load_instruct_mix(cfg.n_instruct_mix, cfg.instruct_dataset)
    examples = [{"messages": c["messages"]} for c in calm] + mix

    # Render to text via chat template for SFT on completions.
    texts = [tokenizer.apply_chat_template(e["messages"], tokenize=False)
             for e in examples]
    dataset = Dataset.from_dict({"text": texts})

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

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=max(1, cfg.effective_batch_size),
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir
