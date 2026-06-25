"""SFT LoRA finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8, on the
~1,150 sample mix (650 calm + 500 Dolci instruct). Trains on assistant turns of
full chat conversations.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from .build_sft_dataset import SFT_PATH
from .train_dpo import _fold_system


def train_sft(output_dir: str | None = None,
              cfg: config.SFTTrainConfig | None = None,
              base_model: str = "gemma-3-27b-it") -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or config.SFTTrainConfig()
    output_dir = output_dir or str(config.CHECKPOINT_DIR / "sft-gemma-27b")
    model_id = config.MODEL_REGISTRY[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN or None)
    rows = [json.loads(l) for l in SFT_PATH.read_text().splitlines() if l]
    ds = Dataset.from_list([{"messages": _fold_system(r["messages"])} for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=config.HF_TOKEN or None)

    peft_config = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        layers_to_transform=(list(cfg.lora.layers_to_transform)
                             if cfg.lora.layers_to_transform else None),
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[sft] adapter saved to {output_dir}")
    return Path(output_dir)
