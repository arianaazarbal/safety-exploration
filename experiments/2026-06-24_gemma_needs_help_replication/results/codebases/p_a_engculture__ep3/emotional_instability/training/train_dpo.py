"""DPO finetuning of Gemma-3-27B-it (Section 4.1; Appendix E, I).

1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 (alpha 64) on all attention+MLP
projections, effective batch size 8. Supports the Appendix I layer-ablation by
restricting LoRA to a layer range via ``--layers a b``.

Usage:
    python -m emotional_instability.training.train_dpo
    python -m emotional_instability.training.train_dpo --layers 30 35   # layer ablation
"""
from __future__ import annotations

import argparse

from ..config import load_config
from .hyperparams import dpo_from_config


def _grad_accum(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(config, layers: tuple[int, int] | None = None, per_device_batch: int = 1,
              output_subdir: str | None = None) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = dpo_from_config(config)
    spec = config.model_by_name(config.finetune_base)
    data_path = str(config.output_path("training", "dpo_pairs.jsonl"))

    tag = output_subdir or ("dpo" if layers is None else f"dpo_L{layers[0]}-{layers[1]}")
    out_dir = str(config.output_path("checkpoints", tag).parent / tag)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    layers_to_transform = list(range(layers[0], layers[1])) if layers else None
    peft_config = LoraConfig(
        r=hp.lora_rank, lora_alpha=hp.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=hp.target_modules,
        layers_to_transform=layers_to_transform,
    )

    # TRL expects columns: prompt (chat list), chosen, rejected.
    ds = load_dataset("json", data_files=data_path, split="train")

    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=hp.epochs,
        learning_rate=hp.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(hp.effective_batch_size, per_device_batch),
        beta=hp.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[dpo] adapter saved -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it")
    ap.add_argument("--config", default=None)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    help="restrict LoRA to layer range [a, b) for Appendix I ablation")
    ap.add_argument("--per-device-batch", type=int, default=1)
    args = ap.parse_args()
    config = load_config(args.config)
    train_dpo(config, tuple(args.layers) if args.layers else None, args.per_device_batch)


if __name__ == "__main__":
    main()
