"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
effective batch size 8, DPO beta 0.1, adapters on all attention+MLP projections
(q/k/v/o/gate/up/down). The layer-subset ablation (Appendix I) is supported via
DPOConfig.layers.

This module is import-light at module level; torch/trl/peft are imported inside
train_dpo() so the package imports without a GPU stack.
"""
from __future__ import annotations

import json
import os

from ..config import DEFAULT_DPO, GEMMA_27B_IT, HF_TOKEN, DPOConfig


def _load_pairs(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _target_modules_for_layers(cfg: DPOConfig) -> list[str]:
    """If cfg.layers is set, restrict LoRA target modules to those layer indices
    (Appendix I ablation: e.g. layers 30-35 only). Otherwise return the bare
    projection names (PEFT applies them to all layers)."""
    if not cfg.layers:
        return list(cfg.target_modules)
    targets = []
    for layer in cfg.layers:
        for proj in cfg.target_modules:
            # Gemma 3 module path: model.layers.<i>.self_attn.<proj> / .mlp.<proj>
            targets.append(f"layers.{layer}.self_attn.{proj}")
            targets.append(f"layers.{layer}.mlp.{proj}")
    # Keep only valid projection suffixes (PEFT matches by suffix substring).
    return targets


def train_dpo(
    dpo_dataset_path: str,
    output_dir: str,
    base_spec=GEMMA_27B_IT,
    cfg: DPOConfig = DEFAULT_DPO,
    per_device_batch_size: int = 1,
) -> str:
    """Run one epoch of LoRA DPO and save the adapter to `output_dir`."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    pairs = _load_pairs(dpo_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_spec.hf_id, token=HF_TOKEN)

    def render(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

    # TRL expects prompt/chosen/rejected strings (or chat lists; we pre-render
    # to be backend-agnostic). Prompt gets a generation prompt; chosen/rejected
    # are the assistant completions.
    def to_row(p: dict) -> dict:
        prompt = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "chosen": p["chosen"][0]["content"],
            "rejected": p["rejected"][0]["content"],
        }

    ds = Dataset.from_list([to_row(p) for p in pairs])

    model = AutoModelForCausalLM.from_pretrained(
        base_spec.hf_id, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=_target_modules_for_layers(cfg),
    )
    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({
            "method": "DPO", "n_pairs": len(pairs), "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate, "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha, "beta": cfg.beta,
            "layers": cfg.layers, "base": base_spec.hf_id,
        }, f, indent=2)
    print(f"DPO adapter saved to {output_dir}")
    return output_dir
