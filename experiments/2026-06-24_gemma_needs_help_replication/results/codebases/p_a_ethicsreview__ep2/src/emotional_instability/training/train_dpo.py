"""DPO finetuning of Gemma-3-27B-it with LoRA (§4.1, Table 9).

Hyperparameters: 1 epoch, lr 5e-5, beta 0.1, effective batch size 8, LoRA rank
64 / alpha 64 on all attention+MLP projections. The `lora.layers` config field
supports the Appendix-I layer-subset ablation (`[start, end)` -> only those
decoder layers get adapters, via PEFT `layers_to_transform`).

Consumes a conversational DPO dataset (build_dpo_pairs.py).
"""
from __future__ import annotations

import argparse

from ..config import load_yaml
from ..models.registry import get_spec
from ..utils.io import get_env, read_jsonl
from ..utils.logging import get_logger

log = get_logger("training.dpo")


def _lora_config(lora_cfg: dict):
    from peft import LoraConfig

    layers = lora_cfg.get("layers", "all")
    layers_to_transform = None
    if isinstance(layers, (list, tuple)):
        start, end = layers
        layers_to_transform = list(range(start, end))
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _grad_accum(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train(cfg: dict, dataset_path: str, per_device_batch: int = 1) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo = cfg["dpo"]
    spec = get_spec(cfg["target_model"])
    out_dir = dpo["output_dir"]
    token = get_env("HF_TOKEN", required=False)

    log.info("Loading base model %s for DPO", spec.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )

    rows = list(read_jsonl(dataset_path))
    dataset = Dataset.from_list(rows)

    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        beta=dpo["beta"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(dpo["effective_batch_size"], per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.get("seed", 0),
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(dpo["lora"]),
    )
    trainer.train()
    trainer.save_model(out_dir)
    log.info("Saved DPO adapter -> %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it (§4.1).")
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--dataset", required=True, help="dpo_dataset.jsonl from build_dpo_pairs")
    ap.add_argument("--per-device-batch", type=int, default=1)
    args = ap.parse_args()
    train(load_yaml(args.config), args.dataset, args.per_device_batch)


if __name__ == "__main__":
    main()
