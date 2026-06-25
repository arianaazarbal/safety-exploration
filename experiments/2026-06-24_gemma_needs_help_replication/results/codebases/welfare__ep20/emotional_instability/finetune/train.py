"""LoRA finetuning of Gemma-3-27B-it: DPO or SFT (Section 4.1, Table 9).

Usage:
    python -m emotional_instability.finetune.train --method dpo
    python -m emotional_instability.finetune.train --method sft
    python -m emotional_instability.finetune.train --method dpo --layers 30-35   # Appendix I ablation

Hyperparameters (Table 9):
    DPO : 280 pairs, 1 epoch, lr 5e-5, LoRA r=64 alpha=64, beta=0.1, eff. batch 8
    SFT : 1,150 samples, 2 epochs, lr 1e-4, LoRA r=64 alpha=128, eff. batch 8
    LoRA target modules: q,k,v,o,gate,up,down proj (all layers by default).

Adapters are written to `adapters/<method>[_L<layers>]`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config


def _parse_layers(spec: str | None) -> list[int] | None:
    """'30-35' -> [30,31,32,33,34], '40' -> [40], None -> None (all layers)."""
    if not spec:
        return None
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi)))
    return [int(spec)]


def _lora_config(rank, alpha, target_modules, layers):
    from peft import LoraConfig
    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
        layers_to_transform=layers,            # None => all layers
    )


def _grad_accum(effective_batch_size: int, per_device: int) -> int:
    return max(1, effective_batch_size // per_device)


def train_dpo(cfg, layers, per_device, out_dir):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    fc = cfg["finetune"]
    d = fc["dpo"]
    model_id = fc["base_model"]
    data_path = config.resolve_path(cfg, "data_dir") / "dpo_pairs.jsonl"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=d["epochs"],
        learning_rate=d["learning_rate"],
        beta=d["beta"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=_grad_accum(d["effective_batch_size"], per_device),
        bf16=True, logging_steps=5, save_strategy="no",
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tok,
        peft_config=_lora_config(d["lora_rank"], d["lora_alpha"],
                                 fc["lora_target_modules"], layers),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train] saved DPO adapter -> {out_dir}")


def train_sft(cfg, layers, per_device, out_dir):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    fc = cfg["finetune"]
    s = fc["sft"]
    model_id = fc["base_model"]
    data_path = config.resolve_path(cfg, "data_dir") / "sft_dataset.jsonl"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=s["epochs"],
        learning_rate=s["learning_rate"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=_grad_accum(s["effective_batch_size"], per_device),
        bf16=True, logging_steps=5, save_strategy="no",
        max_length=cfg["sampling"]["max_tokens"],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, processing_class=tok,
        peft_config=_lora_config(s["lora_rank"], s["lora_alpha"],
                                 fc["lora_target_modules"], layers),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train] saved SFT adapter -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--layers", default=None,
                    help="LoRA layer subset for the Appendix I ablation, e.g. 30-35")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = config.load_config(args.config)
    layers = _parse_layers(args.layers)
    adapters_dir = config.resolve_path(cfg, "adapters_dir")
    tag = args.method + (f"_L{args.layers}" if args.layers else "")
    out_dir = Path(adapters_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "dpo":
        train_dpo(cfg, layers, args.per_device_batch, out_dir)
    else:
        train_sft(cfg, layers, args.per_device_batch, out_dir)


if __name__ == "__main__":
    main()
