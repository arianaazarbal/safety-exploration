"""DPO and SFT LoRA finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters follow Table 9:
  DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1, eff. batch 8.
  SFT: 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, eff. batch 8.
LoRA is applied to all attention + MLP projections (q/k/v/o/gate/up/down).

Both produce a LoRA adapter under data/adapters/, which can be passed to
ei.run_eval via --adapter for re-evaluation.

This module is import-light at module scope; the heavy ML stack is imported inside
the train functions so the rest of the package works without a GPU.
"""
from __future__ import annotations

import argparse

import config
from ..utils import read_jsonl
from .build_dataset import DPO_PATH, SFT_PATH


def _grad_accum(per_device_bs: int, effective_bs: int) -> int:
    return max(1, effective_bs // per_device_bs)


def train_dpo(output_dir: str, per_device_bs: int = 1, seed: int = 0) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base = config.MODEL_REGISTRY[config.FINETUNE_BASE_MODEL].model_id
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")

    rows = read_jsonl(DPO_PATH)
    if not rows:
        raise SystemExit("No DPO pairs found. Run mitigation.build_dataset first.")
    ds = Dataset.from_list(rows)               # conversational prompt/chosen/rejected

    peft_cfg = LoraConfig(
        r=config.DPO.lora_rank, lora_alpha=config.DPO.lora_alpha,
        target_modules=config.LORA_TARGET_MODULES, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM")

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_grad_accum(per_device_bs,
                                                config.DPO.effective_batch_size),
        bf16=True, logging_steps=10, save_strategy="epoch", seed=seed,
        max_length=2048, max_prompt_length=1536,
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir


def train_sft(output_dir: str, per_device_bs: int = 1, seed: int = 0) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base = config.MODEL_REGISTRY[config.FINETUNE_BASE_MODEL].model_id
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")

    rows = read_jsonl(SFT_PATH)
    if not rows:
        raise SystemExit("No SFT data found. Run mitigation.build_dataset first.")
    ds = Dataset.from_list(rows)               # conversational 'messages'

    peft_cfg = LoraConfig(
        r=config.SFT.lora_rank, lora_alpha=config.SFT.lora_alpha,
        target_modules=config.LORA_TARGET_MODULES, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM")

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_grad_accum(per_device_bs,
                                                config.SFT.effective_batch_size),
        bf16=True, logging_steps=10, save_strategy="epoch", seed=seed,
        max_length=2048, packing=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir


def main() -> None:
    p = argparse.ArgumentParser(description="DPO / SFT LoRA finetuning of Gemma")
    p.add_argument("--method", choices=["dpo", "sft"], required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--per-device-bs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.output_dir or str(config.ADAPTERS_DIR / args.method)
    if args.method == "dpo":
        train_dpo(out, args.per_device_bs, args.seed)
    else:
        train_sft(out, args.per_device_bs, args.seed)


if __name__ == "__main__":
    main()
