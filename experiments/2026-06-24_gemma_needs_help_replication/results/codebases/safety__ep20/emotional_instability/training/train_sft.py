"""SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E / Table 9).

LoRA rank-64 (alpha 128) on all attention + MLP projections; 2 epochs; lr 1e-4;
effective batch size 8. Trains on 650 calm responses + 500 Dolci-Instruct-SFT
samples. In the paper SFT fails to reduce frustration; we include it for the
SFT-vs-DPO comparison (Figure 5).
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .. import config
from .build_dataset import load_jsonl


def _grad_accum(cfg) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


def train_sft(
    examples: List[dict],
    output_dir: str,
    cfg: config.SFTConfig = config.SFT,
    runtime: Optional[config.RuntimeConfig] = None,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    runtime = runtime or config.RUNTIME
    dtype = getattr(torch, runtime.dtype)
    token = config.get_key(config.HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, token=token, torch_dtype=dtype, device_map=runtime.device_map,
    )

    dataset = Dataset.from_list(examples)  # each row: {"messages": [...]}

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(config.LORA_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        max_length=cfg.max_length,
        bf16=(runtime.dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft] saved LoRA adapter -> {output_dir}")
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/sft_dataset.jsonl")
    ap.add_argument("--output", default="results/gemma-3-27b-sft")
    args = ap.parse_args()
    train_sft(load_jsonl(args.data), args.output)


if __name__ == "__main__":
    main()
