"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

Hyperparameters: 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1, effective
batch size 8, adapters on all attention + MLP projections.

Supports the Appendix I layer-subset ablation via --layers (e.g. "30-35"):
restricts the LoRA adapters to a contiguous range of decoder layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from ..config import load_config


def _parse_layers(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    if "-" in arg:
        lo, hi = arg.split("-")
        return list(range(int(lo), int(hi)))  # [lo, hi)
    return [int(x) for x in arg.split(",")]


def train(dataset_path: str, output_dir: str, *, layers: list[int] | None = None,
          batch_size: int = 1) -> str:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = load_config()
    tcfg = cfg.train
    spec = cfg.participant("gemma-3-27b-it")

    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora = LoraConfig(
        r=tcfg["lora"]["rank"],
        lora_alpha=tcfg["dpo"]["lora_alpha"],
        target_modules=tcfg["lora"]["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=layers,            # None => all layers
        layers_pattern="layers" if layers else None,
    )

    grad_accum = max(1, tcfg["dpo"]["effective_batch_size"] // batch_size)
    dpo_cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=tcfg["dpo"]["epochs"],
        learning_rate=tcfg["dpo"]["learning_rate"],
        beta=tcfg["dpo"]["beta"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(cfg.paths["data_dir"] / "dpo_dataset.jsonl"))
    ap.add_argument("--output", default=str(cfg.paths["models_dir"] / "dpo"))
    ap.add_argument("--layers", default=None,
                    help="restrict LoRA to a layer range, e.g. '30-35' (Appendix I)")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    cfg.ensure_dirs()
    out = args.output
    if args.layers:
        out = f"{out}_layers{args.layers}"
    train(args.dataset, out, layers=_parse_layers(args.layers), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
