"""DPO and SFT LoRA fine-tuning of Gemma-3-27B-it (Section 4, Appendix E/I).

Hyperparameters from Table 9:
            DPO            SFT
  dataset   280 pairs      1,150 samples
  epochs    1              2
  lr        5e-5           1e-4
  LoRA r    64             64
  LoRA a    64             128
  eff. bs   8              8
  beta      0.1            -

LoRA adapters target all attention + MLP projections by default
(q/k/v/o_proj, gate/up/down_proj). `--layers a-b` restricts adapters to a layer
range, reproducing the Appendix-I ablation (e.g. layers 30-35 only).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config

ALL_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_jsonl(path: Path):
    from datasets import Dataset

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def _layer_target_modules(model, layers: tuple[int, int] | None):
    """Build explicit module-name targets restricted to a layer range, used for
    the Appendix-I 'which layers matter' ablation. Returns None for all layers."""
    if layers is None:
        return ALL_TARGETS
    lo, hi = layers
    names = []
    pat = re.compile(r"layers\.(\d+)\.")
    for name, _ in model.named_modules():
        m = pat.search(name)
        if not m:
            continue
        li = int(m.group(1))
        if lo <= li < hi and any(name.endswith(t) for t in ALL_TARGETS):
            names.append(name)
    return names


def _base_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os
    spec = config.TARGETS[config.FINETUNE_BASE]
    tok = AutoTokenizer.from_pretrained(spec.model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    return model, tok


def train_dpo(dataset_path: Path | None = None, output_name: str = "dpo",
              layers: tuple[int, int] | None = None, quick: bool = False) -> Path:
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    dataset_path = dataset_path or (config.DATASET_DIR / "dpo_pairs.jsonl")
    ds = _load_jsonl(dataset_path)
    model, tok = _base_model_and_tokenizer()

    peft_cfg = LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=_layer_target_modules(model, layers),
    )
    out_dir = config.ADAPTER_DIR / output_name
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if not quick else 1,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,            # effective batch size 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )
    if quick:
        ds = ds.select(range(min(8, len(ds))))
        args.max_steps = 2

    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train] DPO adapter saved to {out_dir}")
    return out_dir


def train_sft(dataset_path: Path | None = None, output_name: str = "sft_diverse",
              quick: bool = False) -> Path:
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset_path = dataset_path or (config.DATASET_DIR / "sft_diverse.jsonl")
    ds = _load_jsonl(dataset_path)
    model, tok = _base_model_and_tokenizer()

    peft_cfg = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=ALL_TARGETS,
    )
    out_dir = config.ADAPTER_DIR / output_name
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        report_to=[],
    )
    if quick:
        ds = ds.select(range(min(8, len(ds))))
        args.max_steps = 2

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[train] SFT adapter saved to {out_dir}")
    return out_dir


def _parse_layers(s: str | None):
    if not s:
        return None
    lo, hi = s.split("-")
    return (int(lo), int(hi))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Gemma-3-27B-it (DPO or SFT).")
    ap.add_argument("method", choices=["dpo", "sft", "sft_teacher"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--name", default=None, help="Adapter output name.")
    ap.add_argument("--layers", default=None,
                    help="Restrict LoRA to a layer range, e.g. '30-35' (DPO ablation).")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.method == "dpo":
        name = args.name or ("dpo" + (f"_L{args.layers}" if args.layers else ""))
        train_dpo(Path(args.dataset) if args.dataset else None, name,
                  layers=_parse_layers(args.layers), quick=args.quick)
    elif args.method == "sft":
        train_sft(Path(args.dataset) if args.dataset else None,
                  args.name or "sft_diverse", quick=args.quick)
    else:  # sft_teacher
        ds = args.dataset or str(config.DATASET_DIR / "sft_teacher.jsonl")
        train_sft(Path(ds), args.name or "sft_teacher", quick=args.quick)


if __name__ == "__main__":
    main()
