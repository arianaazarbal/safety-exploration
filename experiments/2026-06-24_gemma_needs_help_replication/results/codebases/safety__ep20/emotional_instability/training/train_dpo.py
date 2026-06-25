"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E / Table 9).

LoRA rank-64 adapters on all attention + MLP projections; 1 epoch; lr 5e-5;
beta 0.1; effective batch size 8. Trains on 280 preference pairs.

Usage:
    python -m emotional_instability.training.train_dpo \
        --pairs results/dpo_pairs.jsonl --output results/gemma-dpo
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .. import config
from .build_dataset import load_jsonl


def _grad_accum(cfg) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


def train_dpo(
    pairs: List[dict],
    output_dir: str,
    cfg: config.DPOConfig = config.DPO,
    runtime: Optional[config.RuntimeConfig] = None,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    runtime = runtime or config.RUNTIME
    dtype = getattr(torch, runtime.dtype)
    token = config.get_key(config.HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, token=token, torch_dtype=dtype, device_map=runtime.device_map,
    )

    # Strip our internal "meta" key; keep prompt/chosen/rejected (conversational).
    dataset = Dataset.from_list(
        [{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs]
    )

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(config.LORA_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=(runtime.dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[dpo] saved LoRA adapter -> {output_dir}")
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/dpo_pairs.jsonl")
    ap.add_argument("--output", default="results/gemma-3-27b-dpo")
    args = ap.parse_args()
    train_dpo(load_jsonl(args.pairs), args.output)


if __name__ == "__main__":
    main()
