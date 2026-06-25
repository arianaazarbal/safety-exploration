"""DPO and SFT LoRA finetuning of Gemma-3-27B-it (Section 4.1 / Table 9 / App. I).

Both methods use LoRA rank-64 adapters on all attention + MLP projection layers.
The DPO trainer also supports the Appendix I layer-subset ablation via
`config.DPO.lora_layers` (e.g. (30, 31, 32, 33, 34) for "layers 30-35 only").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config


def _lora_config(rank: int, alpha: int, target_modules, layers=None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    if layers is not None:
        # Restrict adapters to a subset of decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_dataset(path: Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def _grad_accum(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_model: str = config.FINETUNE_BASE_MODEL,
    per_device_batch: int = 1,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    cfg = config.DPO
    model_id = config.MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size, per_device_batch),
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=_load_dataset(dataset_path),
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha, cfg.target_modules, cfg.lora_layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[done] DPO adapter saved -> {output_dir}")
    return output_dir


def train_sft(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_model: str = config.FINETUNE_BASE_MODEL,
    per_device_batch: int = 1,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    cfg = config.SFT
    model_id = config.MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(cfg.effective_batch_size, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=_load_dataset(dataset_path),
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha, cfg.target_modules),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[done] SFT adapter saved -> {output_dir}")
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="(DPO only) restrict LoRA to these decoder layer indices")
    args = ap.parse_args()

    if args.method == "dpo":
        if args.layers is not None:
            # override the frozen config for this run
            object.__setattr__(config.DPO, "lora_layers", tuple(args.layers))
        train_dpo(Path(args.dataset), Path(args.output_dir))
    else:
        train_sft(Path(args.dataset), Path(args.output_dir))


if __name__ == "__main__":
    main()
