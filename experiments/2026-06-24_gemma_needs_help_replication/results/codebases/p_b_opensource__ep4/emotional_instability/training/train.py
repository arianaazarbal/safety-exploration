"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

Hyperparameters are pulled from `config.DPO_CONFIG` / `config.SFT_CONFIG`:

           DPO            SFT
  epochs    1              2
  lr        5e-5           1e-4
  LoRA r    64             64
  LoRA a    64             128
  beta      0.1            -
  batch     8 (effective)  8 (effective)
  LoRA on q/k/v/o/gate/up/down projections, all layers.

`--layers a b c ...` restricts the adapters to specific decoder layers, used by
the Appendix I layer-ablation study (e.g. `--layers 30 31 32 33 34` for the
"30-35 only" condition). Effective batch size 8 is realised as
per_device_batch_size x grad_accum; tune the split to your hardware (see
DESIGN.md — 27B LoRA training needs a large/multi GPU).
"""

from __future__ import annotations

import argparse
import json

from ..config import DPO_CONFIG, LoRAConfig, SFT_CONFIG, TrainConfig

GEMMA_IT = "google/gemma-3-27b-it"


def _load_jsonl(path: str):
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _peft_config(lora: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        # Appendix I: restrict adapters to a subset of decoder layers.
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _batch_split(effective: int, per_device: int) -> tuple[int, int]:
    grad_accum = max(1, effective // per_device)
    return per_device, grad_accum


def train_dpo(
    dataset_path: str,
    output_dir: str,
    cfg: TrainConfig = DPO_CONFIG,
    per_device_batch_size: int = 1,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    ds = _load_jsonl(dataset_path)
    tok = AutoTokenizer.from_pretrained(GEMMA_IT)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT, torch_dtype=torch.bfloat16, device_map="auto"
    )
    pdb, ga = _batch_split(cfg.effective_batch_size, per_device_batch_size)
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=pdb,
        gradient_accumulation_steps=ga,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def train_sft(
    dataset_path: str,
    output_dir: str,
    cfg: TrainConfig = SFT_CONFIG,
    per_device_batch_size: int = 1,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ds = _load_jsonl(dataset_path)
    tok = AutoTokenizer.from_pretrained(GEMMA_IT)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT, torch_dtype=torch.bfloat16, device_map="auto"
    )
    pdb, ga = _batch_split(cfg.effective_batch_size, per_device_batch_size)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=pdb,
        gradient_accumulation_steps=ga,
        max_seq_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # Train on the assistant turn(s) only; the conversational collator masks
        # prompt tokens when the chat template marks assistant spans.
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="DPO/SFT LoRA finetuning of Gemma-3-27B-it")
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--dataset", required=True, help="JSONL produced by build_datasets")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Restrict LoRA to these decoder layers (Appendix I).")
    args = ap.parse_args(argv)

    base = DPO_CONFIG if args.method == "dpo" else SFT_CONFIG
    if args.layers is not None:
        lora = LoRAConfig(
            r=base.lora.r, alpha=base.lora.alpha, dropout=base.lora.dropout,
            target_modules=base.lora.target_modules,
            layers_to_transform=tuple(args.layers),
        )
        cfg = TrainConfig(
            method=base.method, epochs=base.epochs, learning_rate=base.learning_rate,
            effective_batch_size=base.effective_batch_size, lora=lora,
            dpo_beta=base.dpo_beta, max_seq_len=base.max_seq_len,
        )
    else:
        cfg = base

    if args.method == "dpo":
        train_dpo(args.dataset, args.output_dir, cfg, args.per_device_batch_size)
    else:
        train_sft(args.dataset, args.output_dir, cfg, args.per_device_batch_size)


if __name__ == "__main__":
    main()
