"""SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

LoRA rank-64 / alpha-128 on all attention+MLP projections, 2 epochs, lr 1e-4,
effective batch size 8. Trains on the chat-format dataset from
``data/sft_<variant>.jsonl`` and writes a LoRA adapter to
``adapters/sft_<variant>`` (registered as gemma-3-27b-sft-<variant>).

The paper finds SFT ineffective at reducing distress (the "teacher" variant
even increases it); this reproduces both variants.
"""

from __future__ import annotations

import argparse

import config
from ..utils.io import read_jsonl


def train(variant: str = "diverse", base_model: str = "gemma-3-27b-it",
          dtype: str = "bfloat16"):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    cfg = config.SFT_CFG
    model_id = config.TARGET_MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    )

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )

    examples = read_jsonl(config.DATA_DIR / f"sft_{variant}.jsonl")
    if not examples:
        raise SystemExit(f"[train_sft] need data/sft_{variant}.jsonl "
                         "(training.build_sft_data)")
    ds = Dataset.from_list([{"messages": e["messages"]} for e in examples])

    output_dir = str(config.ADAPTER_DIR / f"sft_{variant}")
    train_args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_sft] saved adapter (sft_{variant}) -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()
    train(args.variant)
